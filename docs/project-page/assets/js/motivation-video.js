(() => {
  "use strict";
  const video = document.querySelector("[data-motivation-video]");
  if (!video) return;
  const config = window.VLA_DELTA_SITE?.motivationVideo || {};
  if (config.src && video.getAttribute("src") !== config.src) video.src = config.src;
  if (config.poster) video.poster = config.poster;
  document.querySelectorAll("[data-motivation-download]").forEach(link => {
    if (config.src) link.href = config.src;
  });
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let inView = false;
  let started = false;
  let resumeOnEntry = false;
  const play = () => {
    started = true;
    resumeOnEntry = false;
    const attempt = video.play();
    if (attempt) attempt.catch(() => { /* The native play button remains available. */ });
  };
  const syncPlayback = () => {
    if (!inView || document.hidden) {
      if (!video.paused && !video.ended) {
        resumeOnEntry = true;
        video.pause();
      }
      return;
    }
    if (!motion.matches && !video.ended && (!started || resumeOnEntry)) play();
  };
  video.addEventListener("play", () => { started = true; });
  video.addEventListener("ended", () => { resumeOnEntry = false; });
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(entries => {
      inView = entries[0].isIntersecting && entries[0].intersectionRatio >= 0.45;
      syncPlayback();
    }, { threshold: [0, 0.45], rootMargin: "-90px 0px 0px 0px" }).observe(video);
  }
  document.addEventListener("visibilitychange", syncPlayback);
  motion.addEventListener("change", () => {
    if (motion.matches) { video.pause(); resumeOnEntry = false; }
    else syncPlayback();
  });
})();
