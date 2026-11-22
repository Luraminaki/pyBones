// An image fed with slices: one request at a time (a slice asked for meanwhile is loaded right
// after, the latest winning, so dragging a slider never piles renders up on the server), each
// shown through a temporary blob: URL revoked once loaded, so the browser keeps no slice.

import { fetchSlice } from './api.js';
import { toast } from './ui.js';

/**
 * Point an image at a URL and wait until it has loaded. The load event fires whether the image
 * is displayed or not (`decode()` refuses an image in a hidden tab).
 * @param {HTMLImageElement} image The image.
 * @param {string} url Its new source.
 * @returns {Promise<void>} Resolves once loaded.
 */
function loaded(image, url) {
  return new Promise((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error('the image could not be read'));
    image.src = url;
  });
}

/**
 * Feed an image with slices of segmentations.
 * @param {HTMLImageElement} image The image.
 * @param {function(boolean): void} busy Told when loading starts (true) and when the last
 *   request is over (false).
 * @returns {function(string, number, (number|string)): void} Shows slice `z` of a segmentation
 *   at a scale (the latest call wins).
 */
export function sliceLoader(image, busy) {
  let inFlight = false;
  let next = null; // the slice asked for while another one was loading

  const run = async (target) => {
    inFlight = true;
    busy(true);
    let url = null;
    try {
      url = URL.createObjectURL(await fetchSlice(target.id, target.z, target.scale));
      image.alt = `Slice ${target.z} of ${target.id} with its labels overlaid`;
      await loaded(image, url);
    } catch (err) {
      // A newer slice is waiting: this one no longer matters.
      if (!next) toast(`Slice ${target.z} could not be shown: ${err.message}`, { error: true });
    } finally {
      if (url) URL.revokeObjectURL(url);
      inFlight = false;
      if (next) {
        const target = next;
        next = null;
        void run(target);
      } else {
        busy(false);
      }
    }
  };

  return (id, z, scale) => {
    const target = { id, z, scale };
    if (inFlight) next = target;
    else void run(target);
  };
}
