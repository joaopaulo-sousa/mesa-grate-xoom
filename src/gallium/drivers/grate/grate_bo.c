/*
 * Copyright © 2012, 2013 Thierry Reding
 * Copyright © 2013 Erik Faye-Lund
 * Copyright © 2014 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include "util/hash_table.h"
#include "util/u_dynarray.h"
#include <util/os_mman.h>
#include <util/u_atomic.h>
#include <util/macros.h>
#include <util/os_time.h>

#include <xf86drm.h>

#include "drm-uapi/tegra_drm.h"

#include "tegra.h"
#include "grate_common.h"
#include "grate_device.h"
#include "grate_bo.h"

static void grate_bo_free(struct grate_bo *bo)
{
    struct grate_device *drm = bo->drm;

    grate_bo_logd(bo, "");

    if (bo->map)
        munmap(bo->map, bo->size);

    drmCloseBufferHandle(drm->fd, bo->handle);
    
    pthread_mutex_lock(&drm->mapping_lock);
    util_dynarray_foreach(&bo->channel_maps, struct drm_tegra_mapping *, mapping) {
       drm_tegra_channel_unmap(*mapping);
    }
    util_dynarray_fini(&bo->channel_maps);
    pthread_mutex_unlock(&drm->mapping_lock);

    //Clear BO as GEM handles are reused and we are not removing from bo_map
    memset(bo, 0, sizeof(*bo));
}

static void grate_bo_prepare(struct grate_bo *bo)
{
   p_atomic_set(&bo->refcnt, 1);
   util_dynarray_init(&bo->channel_maps, NULL);
}

/*
 * BO cache: a BO the process frees stays allocated (and mapped, and mapped
 * into the channels) for a while, for the next allocation of about its
 * size. A new GEM object costs the kernel a clear and a cache flush of all
 * its pages (CMA): Xorg/glamor creating and dropping a window-sized pixmap
 * per frame of a scrolling WebKit view spent a third of its time there
 * (v7_dma_flush_range, l2c210_flush_range, memset). The reused BO keeps its
 * old contents, which GL leaves undefined anyway.
 */
#define GRATE_BO_CACHE_MAX_SIZE (16u << 20)
#define GRATE_BO_CACHE_MAX_AGE_NS 1000000000ll

/* Free the cached BOs over the size limit or older than the age limit;
 * under bo_map_lock */
static void grate_bo_cache_trim(struct grate_device *drm, int64_t now)
{
   list_for_each_entry_safe(struct grate_bo, bo, &drm->bo_cache, cache_link) {
      if (drm->bo_cache_size <= GRATE_BO_CACHE_MAX_SIZE &&
          now - bo->cache_time < GRATE_BO_CACHE_MAX_AGE_NS)
         break;
      list_del(&bo->cache_link);
      drm->bo_cache_size -= bo->size;
      grate_bo_free(bo);
   }
}

/* Whether the GPU is done with the BO, without waiting */
static bool grate_bo_idle(struct grate_bo *bo)
{
   bool idle = true;

   pthread_mutex_lock(&bo->drm->fence_lock);
   for (unsigned i = 0; i < ARRAY_SIZE(bo->fence) && idle; i++) {
      if (!bo->fence[i].drm)
         continue;
      if (drm_tegra_fence_wait(&bo->fence[i], 0))
         idle = false;
      else
         bo->fence[i].drm = NULL;
   }
   pthread_mutex_unlock(&bo->drm->fence_lock);
   return idle;
}

/* A cached BO of the flags and of size .. size * 5 / 4 the GPU is done
 * with, most recently freed first; NULL when none */
static struct grate_bo *grate_bo_cache_get(struct grate_device *drm, uint32_t size,
                                           uint32_t flags)
{
   struct grate_bo *ret = NULL;

   pthread_mutex_lock(&drm->bo_map_lock);
   grate_bo_cache_trim(drm, os_time_get_nano());
   list_for_each_entry_rev(struct grate_bo, bo, &drm->bo_cache, cache_link) {
      if (bo->flags != flags || bo->size < size || bo->size > size + size / 4 ||
          !grate_bo_idle(bo))
         continue;
      list_del(&bo->cache_link);
      drm->bo_cache_size -= bo->size;
      p_atomic_set(&bo->refcnt, 1);
      ret = bo;
      break;
   }
   pthread_mutex_unlock(&drm->bo_map_lock);
   return ret;
}

void grate_bo_cache_fini(struct grate_device *drm)
{
   pthread_mutex_lock(&drm->bo_map_lock);
   list_for_each_entry_safe(struct grate_bo, bo, &drm->bo_cache, cache_link) {
      list_del(&bo->cache_link);
      grate_bo_free(bo);
   }
   drm->bo_cache_size = 0;
   pthread_mutex_unlock(&drm->bo_map_lock);
}

struct grate_bo *grate_bo_alloc(struct grate_device *drm, uint32_t size, uint32_t flags)
{
    struct drm_tegra_gem_create args;
    struct grate_bo *bo;
    int err;

    if (!drm || size == 0)
        return NULL;

    size = align(size, 4096);
    bo = grate_bo_cache_get(drm, size, flags);
    if (bo)
        return bo;

    memset(&args, 0, sizeof(args));
    args.flags = flags;
    args.size = size;

    err = drmCommandWriteRead(drm->fd, DRM_TEGRA_GEM_CREATE, &args,
                              sizeof(args));
    if (err < 0) {
        grate_loge("failed size %u bytes flags 0x%08X err %d (%s)\n",
           size, flags, err, strerror(-err));
        return NULL;
    }
    
    pthread_mutex_lock(&drm->bo_map_lock);
    bo = grate_lookup_bo(drm, args.handle);
    pthread_mutex_unlock(&drm->bo_map_lock);
    if (!bo) {
       grate_loge("failed to allocate bo slot in the bo_map array");
       drmCloseBufferHandle(drm->fd, bo->handle);
       return NULL;
    }
    
    /* Fresh handle */
    assert(!memcmp(bo, &((struct grate_bo){}), sizeof(*bo)));

    bo->handle = args.handle;
    bo->drm = drm;
    bo->size = args.size;
    bo->flags = flags;

    grate_bo_prepare(bo);

    grate_bo_logd(bo, "");

    return bo;
}

struct grate_bo *grate_bo_ref(struct grate_bo *bo)
{
   if (bo) {
      grate_bo_loge(bo, "");
      ASSERTED int count = p_atomic_inc_return(&bo->refcnt);
      assert(count != 1);
   }

    return bo;
}

/*
 * Record a job that used the BO at submit time (grate_stream_flush): it
 * replaces the earlier one on its syncpoint, takes a free entry otherwise;
 * with all of them taken by other syncpoints, the first one is waited for.
 */
void grate_bo_add_fence(struct grate_bo *bo, const struct drm_tegra_fence *fence)
{
   int free = -1;

   pthread_mutex_lock(&bo->drm->fence_lock);
   for (unsigned i = 0; i < ARRAY_SIZE(bo->fence); i++) {
      if (bo->fence[i].drm && bo->fence[i].syncpt == fence->syncpt) {
         bo->fence[i] = *fence;
         pthread_mutex_unlock(&bo->drm->fence_lock);
         return;
      }
      if (!bo->fence[i].drm && free < 0)
         free = i;
   }
   if (free < 0) {
      if (drm_tegra_fence_wait(&bo->fence[0], GRATE_JOB_TIMEOUT_NS))
         fprintf(stderr, "grate: waiting for bo %u failed\n", bo->handle);
      free = 0;
   }
   bo->fence[free] = *fence;
   pthread_mutex_unlock(&bo->drm->fence_lock);
}

/*
 * Wait until the GPU is done with the BO: the jobs that used it are the
 * fences stamped at submit time (grate_bo_add_fence).
 */
int grate_bo_wait(struct grate_bo *bo)
{
   struct drm_tegra_fence fence[GRATE_BO_MAX_FENCES];
   int ret = 0;

   pthread_mutex_lock(&bo->drm->fence_lock);
   memcpy(fence, bo->fence, sizeof(fence));
   pthread_mutex_unlock(&bo->drm->fence_lock);

   for (unsigned i = 0; i < ARRAY_SIZE(fence); i++) {
      if (!fence[i].drm)
         continue;

      int err = drm_tegra_fence_wait(&fence[i], GRATE_JOB_TIMEOUT_NS);
      if (err) {
         ret = err;
         fence[i].drm = NULL;   /* keep the entry */
      }
   }

   /* forget the fences that passed, unless a newer job replaced them */
   pthread_mutex_lock(&bo->drm->fence_lock);
   for (unsigned i = 0; i < ARRAY_SIZE(fence); i++)
      if (fence[i].drm && bo->fence[i].drm &&
          bo->fence[i].syncpt == fence[i].syncpt && bo->fence[i].value == fence[i].value)
         bo->fence[i].drm = NULL;
   pthread_mutex_unlock(&bo->drm->fence_lock);

   if (ret)
      fprintf(stderr, "grate: waiting for bo %u failed: %d\n", bo->handle, ret);

   return ret;
}

void grate_bo_unref(struct grate_bo *bo)
{
   if (!bo)
      return;

   struct grate_device *drm = bo->drm;
   int refcnt = p_atomic_dec_return(&bo->refcnt);
   assert(refcnt >= 0);

   /* Don't return to cache if there are still references */
   if (refcnt)
      return;

   pthread_mutex_lock(&drm->bo_map_lock);

   /* Someone might have imported this BO while we were waiting for the
    * lock, let's make sure it's still not referenced before freeing it.
    */
   if (p_atomic_read(&bo->refcnt) == 0) {
      grate_bo_logd(bo, "");
      if (!bo->shared && bo->size <= GRATE_BO_CACHE_MAX_SIZE / 2) {
         bo->cache_time = os_time_get_nano();
         list_addtail(&bo->cache_link, &drm->bo_cache);
         drm->bo_cache_size += bo->size;
         grate_bo_cache_trim(drm, bo->cache_time);
      } else {
         grate_bo_free(bo);
      }
   }
   
   pthread_mutex_unlock(&drm->bo_map_lock);
}

int
grate_bo_get_handle(struct grate_bo *bo, uint32_t *handle)
{
    if (!bo || !handle)
        return -EINVAL;

    *handle = bo->handle;

    return 0;
}

int grate_bo_map(struct grate_bo *bo, void **ptr)
{
    struct grate_device *drm = bo->drm;

    if (!bo->map) {
        struct drm_tegra_gem_mmap args;
        int err;

        memset(&args, 0, sizeof(args));
        args.handle = bo->handle;

        err = drmCommandWriteRead(drm->fd, DRM_TEGRA_GEM_MMAP, &args,
                                  sizeof(args));
        if (err < 0) {
            grate_bo_loge(bo, "failed get mapping offset err %d (%s)\n",
               err, strerror(-err));
            return -errno;
        }

        bo->offset = args.offset;

        bo->map = os_mmap(NULL, bo->size, PROT_READ | PROT_WRITE, MAP_SHARED,
                           drm->fd, bo->offset);
        if (bo->map == MAP_FAILED) {
            grate_bo_loge(bo, "failed to map offset 0x%llX err %d (%s)\n",
               args.offset, -errno, strerror(errno));
            bo->map = NULL;
            return -errno;
        }

        grate_bo_logd(bo, "success\n");
    }

    if (ptr)
        *ptr = bo->map;

    return 0;
}

/*
 * The CPU mapping stays until the BO is freed (grate_bo_free): mapping a
 * BO again for each transfer cost an ioctl, an mmap and a page fault per
 * page touched
 */
int grate_bo_unmap(struct grate_bo *bo)
{
    if (!bo)
        return -EINVAL;

    return 0;
}

int grate_bo_export(struct grate_bo *bo, uint32_t flags)
{
    int fd, err;

    flags |= DRM_CLOEXEC;
    bo->shared = true;

    err = drmPrimeHandleToFD(bo->drm->fd, bo->handle, flags, &fd);
    if (err < 0) {
        grate_bo_loge(bo, "failed err %d strerror(%s)\n",
            err, strerror(-err));
        return err;
    }

    return fd;
}

static ssize_t fd_get_size(int fd)
{
    ssize_t size, err;

    size = lseek(fd, 0, SEEK_END);
    if (size <= 0)
        return -errno;

    err = lseek(fd, 0, SEEK_SET);
    if (err != 0)
        return -errno;

    return size;
}

struct grate_bo*
grate_bo_import(struct grate_device *drm, int fd)
{
   struct grate_bo *bo;
   ssize_t size;
   int err;
   unsigned gem_handle;

   pthread_mutex_lock(&drm->bo_map_lock);

   err = drmPrimeFDToHandle(drm->fd, fd, &gem_handle);
   if (err < 0) {
      grate_loge("drmPrimeFDToHandle err %d strerror(%s)\n", err, strerror(-err));
      pthread_mutex_unlock(&drm->bo_map_lock);
      return NULL;
   }

   bo = grate_lookup_bo(drm, gem_handle);

   if (!bo->size) {
      size = fd_get_size(fd);
      if (size < 0) {
         grate_loge("fd_get_size err %d strerror(%s)\n", (int) size, strerror((int) -size));
         goto error;
      }
   
      bo->drm = drm;
      bo->size = (uint32_t) size;
      bo->handle = gem_handle;
      bo->shared = true;

      grate_bo_prepare(bo);
   } else {
      bo->shared = true;
      /* bo->refcnt == 0 can happen if the BO
       * was being released but grate_bo_import() acquired the
       * lock before grate_bo_unref(). In that case, refcnt
       * is 0 and we can't use grate_bo_ref() directly, we
       * have to re-initialize the refcnt().
       * Note that grate_bo_unref() checks
       * refcnt value just after acquiring the lock to
       * make sure the object is not freed if grate_bo_import()
       * acquired it in the meantime.
       */
      if (p_atomic_read(&bo->refcnt) == 0)
         p_atomic_set(&bo->refcnt, 1);
      else
         grate_bo_ref(bo);
   }
   pthread_mutex_unlock(&drm->bo_map_lock);
   
   assert(bo->drm != NULL && "post-condition");

   return bo;
   
error:
   memset(bo, 0, sizeof(*bo));
   pthread_mutex_unlock(&drm->bo_map_lock);
   return NULL;
}
