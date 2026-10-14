(() => {
  "use strict";
  const region = document.querySelector("[data-text-reveal]");
  const source = region?.querySelector(".transition-text");
  const maskSupported = window.CSS?.supports("mask-image", "linear-gradient(#000, transparent)") ||
    window.CSS?.supports("-webkit-mask-image", "linear-gradient(#000, transparent)");
  if (!source || !maskSupported || !("requestAnimationFrame" in window)) return;

  const config = window.VLA_DELTA_SITE?.textReveal || {};
  const duration = Number.isFinite(config.durationMs) && config.durationMs > 0 ? config.durationMs : 1000;
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let elapsed = 0;
  let previous = null;
  let frame = null;
  let finished = true;
  let started = false;
  let inView = !("IntersectionObserver" in window);

  const stopFrame = () => {
    if (frame !== null) window.cancelAnimationFrame(frame);
    frame = null;
    previous = null;
  };
  const showAll = () => {
    stopFrame();
    finished = true;
    started = true;
    source.classList.remove("text-reveal-wipe");
    source.style.removeProperty("--reveal-position");
  };
  const tick = now => {
    frame = null;
    if (finished || !inView || document.hidden) {
      previous = null;
      return;
    }
    if (previous !== null) elapsed += now - previous;
    previous = now;
    const progress = Math.min(1, elapsed / duration);
    const eased = progress * progress * (3 - 2 * progress);
    // One soft diagonal mask sweeps the intact paragraph from top-left to bottom-right.
    source.style.setProperty("--reveal-position", `${100 * (1 - eased)}%`);
    if (progress === 1) showAll();
    else frame = window.requestAnimationFrame(tick);
  };
  const syncPlayback = () => {
    if (!finished && inView && !document.hidden) {
      if (frame === null) frame = window.requestAnimationFrame(tick);
    } else stopFrame();
  };
  const play = () => {
    stopFrame();
    started = true;
    source.style.setProperty("--reveal-position", "100%");
    source.classList.add("text-reveal-wipe");
    elapsed = 0;
    finished = false;
    syncPlayback();
  };

  motion.addEventListener("change", () => {
    if (motion.matches) showAll();
  });
  document.addEventListener("visibilitychange", syncPlayback);
  if ("IntersectionObserver" in window) {
    const observer = new window.IntersectionObserver(entries => {
      inView = entries.some(entry => entry.isIntersecting);
      if (inView && !started && !motion.matches) play();
      else syncPlayback();
    }, { rootMargin: "0px 0px -64px 0px", threshold: 0 });
    observer.observe(region);
  }
  if (motion.matches) showAll();
  else if (!("IntersectionObserver" in window)) play();
})();
