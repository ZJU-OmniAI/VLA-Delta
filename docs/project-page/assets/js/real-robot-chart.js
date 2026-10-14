(() => {
  "use strict";
  document.querySelectorAll(".real-robot-chart").forEach(chart => {
    const bars = [...chart.querySelectorAll(".chart-bar")];
    chart.dataset.enhanced = "true";
    let active = null;
    const show = bar => {
      active?.classList.remove("is-active");
      active = bar;
      active?.classList.add("is-active");
    };
    bars.forEach((bar, index) => {
      bar.addEventListener("pointerenter", event => {
        if (event.pointerType !== "touch") show(bar);
      });
      bar.addEventListener("pointerleave", event => {
        if (event.pointerType !== "touch" && active === bar) show(null);
      });
      bar.addEventListener("focus", () => show(bar));
      bar.addEventListener("blur", () => { if (active === bar) show(null); });
      bar.addEventListener("click", () => show(bar));
      bar.addEventListener("keydown", event => {
        if (event.key === "Escape") { show(null); return; }
        let next;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") next = Math.min(index + 1, bars.length - 1);
        if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = Math.max(index - 1, 0);
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = bars.length - 1;
        if (next !== undefined) {
          event.preventDefault();
          bars[next].focus();
          bars[next].scrollIntoView({ block: "nearest", inline: "nearest", behavior: "instant" });
        }
      });
    });
    document.addEventListener("pointerdown", event => {
      if (!bars.some(bar => bar.contains(event.target))) show(null);
    });
  });
})();
