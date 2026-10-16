/* The central place to update confirmed metadata, resource links and media.
 * Keep unavailable URLs null. A null resource is shown as Coming Soon.
 * Use paths relative to index.html for local files; never use /assets/ paths.
 */
window.VLA_DELTA_SITE = {
  title: "VLA-Δ: Learning Physical State Transition for Vision-Language-Action Models",
  tagline: "Don't just imitate actions. Understand change.",
  logo: {
    src: "assets/images/vla-delta-logo-transparent.png",
    width: 1254,
    height: 1254,
    alt: "VLA-Δ logo"
  },
  authors: [
    { given: "Qi", family: "Lu" },
    { given: "Jiarun", family: "Li" },
    { given: "Ce", family: "Zhang" },
    { given: "Miao", family: "Pan" },
    { given: "Yi", family: "Pan" },
    { given: "Xu", family: "Wang" },
    { given: "Wenqi", family: "Zhang" },
    { given: "Xuhong", family: "Zhang" }
  ],
  // Institutions displayed on the homepage; no per-author affiliation markers.
  affiliations: [
    { name: "Zhejiang University", domain: "zju.edu.cn", url: "https://www.zju.edu.cn/" }
  ],
  resources: {
    github: "https://github.com/ZJU-OmniAI/VLA-Delta",
    huggingface: "https://huggingface.co/t0rn0d0",
    paper: "assets/paper/vla-delta.pdf",
    project: "https://zju-omniai.github.io/VLA-Delta/",
    arxiv: "https://arxiv.org/abs/2610.21306"
  },
  citation: {
    entryType: "misc",
    key: "lu2026vladeltalearningphysicalstate",
    provisional: false,
    // Matches the arXiv-provided BibTeX entry.
    fields: {
      title: "VLA-$\\Delta$: Learning Physical State Transition for Vision-Language-Action Models",
      author: "Qi Lu and Jiarun Li and Ce Zhang and Miao Pan and Yi Pan and Xu Wang and Wenqi Zhang and Xuhong Zhang",
      year: "2026",
      eprint: "2610.21306",
      archivePrefix: "arXiv",
      primaryClass: "cs.CV",
      url: "https://arxiv.org/abs/2610.21306"
    }
  },
  figures: {
    // Each base has .png, .webp and -1200.webp variants. Sizes are pixels.
    overview: { base: "assets/images/overview", width: 2605, height: 1121, smallWidth: 1200 },
    architecture: { base: "assets/images/architecture", width: 2870, height: 1026, smallWidth: 1200 },
    robustness: { base: "assets/images/robustness", width: 1701, height: 646, smallWidth: 1200 },
    realFull: { base: "assets/images/real-robot-full", width: 2986, height: 1457, smallWidth: 1200 }
  },
  // First-view section reveals; already visited blocks stay visible.
  scrollReveal: {
    durationMs: 650,
    staggerMs: 80
  },
  // Native webpage animation over six regions of the unchanged PNG, not GIF playback.
  // Play once on first view; 900 ms reveals overlap at 600 ms intervals.
  annotationAnimation: {
    poster: "assets/images/annotation-stages.png",
    headerHeight: 87,
    panelEdges: [0, 489, 975, 1461, 1945, 2427, 2930],
    revealMs: 900,
    staggerMs: 600
  },
  // Follow PPT stages 1–7, the review feedback loop, then the final human audit.
  // These durations describe only the webpage presentation, not processing time.
  pipelineAnimation: {
    revealMs: 850,
    stepMs: 1000,
    feedbackMs: 1600
  },
  // Presentation timing only; this does not represent measured model generation speed.
  textReveal: {
    durationMs: 1000
  },
  // Conceptual 3D failure sequence, not real-robot experimental footage.
  // Plays once on first view. Render timing is defined in prepare-motivation-video.swift.
  motivationVideo: {
    src: "assets/videos/motivation/microwave-bowl-failure.mp4?v=20261014-contact-v1",
    poster: "assets/videos/motivation/microwave-bowl-poster.png?v=20261014-contact-v1"
  },
  // Featured introduction video.
  // Manual playback; keep the matching static HTML fallback in sync.
  heroVideo: {
    src: "Video/introduction.mp4?v=20261016-introduction",
    poster: "Video/introduction.png?v=20261016-introduction",
    title: "VLA-Δ introduction video",
    width: 1920,
    height: 1080,
    description: ""
  },
  // Presentation only: cards travel right at a steady speed; video speed is unchanged.
  videoCarousel: {
    speedPxPerSecond: 28,
    hoverScale: 1.05,
    interactionPauseMs: 1400
  },
  // Demonstrations at the bottom of Real-Robot Evaluation, in source-file order.
  // The original supplied clips remain in assets/images/videos/.
  // Update the matching static HTML fallback when changing these entries.
  videos: [
    {
      "src": "assets/videos/real-robot/1.mp4?v=20261013-no-audio",
      "poster": "assets/videos/real-robot/1-poster.jpg",
      "title": "Obstacle Avoidance and Physical Constraint Compliance",
      "description": "Place the carrots on a rack, making sure they do not touch the cups.",
      "width": 1280,
      "height": 720
    },
    {
      "src": "assets/videos/real-robot/2.mp4?v=20261013-no-audio",
      "poster": "assets/videos/real-robot/2-poster.jpg",
      "title": "Long-Horizon Execution",
      "description": "Put all the items on the tabletop onto the tray.",
      "width": 1280,
      "height": 720
    },
    {
      "src": "assets/videos/real-robot/3.mp4?v=20261013-no-audio",
      "poster": "assets/videos/real-robot/3-poster.jpg",
      "title": "Fine-Grained Manipulation",
      "description": "Put the spoon into the cup.",
      "width": 1280,
      "height": 720
    },
    {
      "src": "assets/videos/real-robot/4.mp4?v=20261013-no-audio",
      "poster": "assets/videos/real-robot/4-poster.jpg",
      "title": "Spatial Reasoning",
      "description": "Place the item closest to the bowl onto the plate.",
      "width": 1280,
      "height": 720
    },
    {
      "src": "assets/videos/real-robot/5.mp4?v=20261013-no-audio",
      "poster": "assets/videos/real-robot/5-poster.jpg",
      "title": "Generalization under Changed Tablecloth",
      "description": "Place the item closest to the bowl onto the plate with a changed tablecloth.",
      "width": 1280,
      "height": 720
    }
  ]
};
