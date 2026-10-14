/* Progressive enhancement: every block is readable before this script runs. */
(() => {
  "use strict";
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (motion.matches || !("IntersectionObserver" in window)) return;

  const config = window.VLA_DELTA_SITE?.scrollReveal || {};
  const duration = Number.isFinite(config.durationMs) && config.durationMs > 0 ? config.durationMs : 650;
  const stagger = Number.isFinite(config.staggerMs) && config.staggerMs >= 0 ? config.staggerMs : 80;
  // Use reading-sized blocks, not entire long sections. Existing demonstration
  // animations keep control of their own children and playback timelines.
  const blocks = Array.from(document.querySelectorAll([
    ".teaser",
    ".section > .section-heading",
    ".section > .overview-copy",
    ".section > .motivation-layout",
    ".section > .paper-figure",
    ".contributions > li",
    ".supervision > article",
    ".section > .data-heading",
    ".metrics > .metric",
    ".section > .result-block",
    ".robustness-layout > *",
    ".section > details",
    ".section > .table-note",
    ".section > .real-demos",
    ".section > .citation-box"
  ].join(",")));
  if (!blocks.length) return;

  const reveal = (block, immediately = false) => {
    if (immediately) block.classList.add("reveal-immediate");
    block.classList.add("is-revealed");
    observer.unobserve(block);
  };
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) reveal(entry.target);
    });
  }, { threshold: 0, rootMargin: "0px 0px -40px 0px" });

  blocks.forEach(block => {
    const rect = block.getBoundingClientRect();
    block.style.setProperty("--scroll-reveal-duration", `${duration}ms`);
    const group = block.parentElement;
    if (group.matches(".contributions, .supervision, .metrics")) {
      // Stagger siblings sharing a row; stacked mobile cards enter individually.
      const earlierInRow = Array.from(group.children).slice(0, Array.from(group.children).indexOf(block))
        .filter(sibling => Math.abs(sibling.getBoundingClientRect().top - rect.top) < 2).length;
      block.style.setProperty("--scroll-reveal-delay", `${Math.min(earlierInRow, 3) * stagger}ms`);
    }
    // Keep the initial screen and restored scroll position immediately readable.
    if (rect.top < window.innerHeight) block.classList.add("is-revealed");
    block.classList.add("scroll-reveal");
    if (!block.classList.contains("is-revealed")) observer.observe(block);
  });

  const revealTarget = target => {
    if (!(target instanceof Element)) return;
    blocks.forEach(block => {
      if (block === target || block.contains(target)) reveal(block, true);
    });
    // A section anchor lands at its heading, not at an invisible section title.
    const heading = target.matches(".section") ? target.querySelector(".section-heading") : null;
    if (heading) reveal(heading, true);
  };
  document.addEventListener("focusin", event => revealTarget(event.target));
  const revealHash = () => {
    try { revealTarget(document.getElementById(decodeURIComponent(location.hash.slice(1)))); }
    catch { /* A malformed hash must not affect normal reading. */ }
  };
  window.addEventListener("hashchange", revealHash);
  window.addEventListener("pageshow", () => {
    blocks.forEach(block => {
      if (block.getBoundingClientRect().top < window.innerHeight) reveal(block, true);
    });
  });
  motion.addEventListener("change", () => {
    if (motion.matches) {
      blocks.forEach(block => reveal(block, true));
      observer.disconnect();
    }
  });
  revealHash();
})();
