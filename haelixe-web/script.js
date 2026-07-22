// Haelixe — shared interaction layer

document.addEventListener('DOMContentLoaded', () => {
  initNav();
  initMobileDrawer();
  initReveal();
  initTensorGrid();
  initClonePill();
});

/* Sticky nav border on scroll */
function initNav() {
  const nav = document.querySelector('.nav');
  if (!nav) return;
  const onScroll = () => {
    nav.classList.toggle('is-scrolled', window.scrollY > 8);
  };
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });
}

/* Mobile drawer toggle */
function initMobileDrawer() {
  const btn = document.querySelector('.nav__menu-btn');
  const drawer = document.querySelector('.mobile-drawer');
  if (!btn || !drawer) return;
  btn.addEventListener('click', () => {
    const open = drawer.classList.toggle('is-open');
    btn.setAttribute('aria-expanded', String(open));
  });
  drawer.querySelectorAll('a').forEach((a) =>
    a.addEventListener('click', () => drawer.classList.remove('is-open'))
  );
}

/* Scroll-triggered reveal */
function initReveal() {
  const items = document.querySelectorAll('.reveal');
  if (!items.length) return;
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
  );
  items.forEach((item) => io.observe(item));
}

/* Signature hero visual: a strided tensor grid performing a
   zero-copy "transpose" — cells re-light in a shuffled order to
   suggest reinterpreted strides rather than moved memory. */
function initTensorGrid() {
  const stage = document.querySelector('.hero__grid-stage');
  if (!stage) return;

  const cols = 8;
  const rows = 4;
  const grid = document.createElement('div');
  grid.className = 'tgrid';
  grid.setAttribute('aria-hidden', 'true');

  const cells = [];
  for (let i = 0; i < cols * rows; i++) {
    const cell = document.createElement('div');
    cell.className = 'tgrid__cell';
    grid.appendChild(cell);
    cells.push(cell);
  }
  stage.appendChild(grid);

  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReduced) {
    // Static, calm state: light a diagonal band once.
    cells.forEach((cell, i) => {
      const r = Math.floor(i / cols);
      const c = i % cols;
      if ((r + c) % 5 === 0) cell.classList.add('is-active');
    });
    return;
  }

  // Build a "stride reinterpretation": walk index order for a
  // transpose of an 8x4 grid without moving underlying data.
  const transposedOrder = [];
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      transposedOrder.push(r * cols + c);
    }
  }

  let step = 0;
  const bandSize = 6;

  function tick() {
    cells.forEach((cell) => cell.classList.remove('is-lit'));
    for (let k = 0; k < bandSize; k++) {
      const idx = transposedOrder[(step + k) % transposedOrder.length];
      cells[idx].classList.add('is-lit');
    }
    step = (step + 1) % transposedOrder.length;
  }

  cells.forEach((cell, i) => {
    if (i % 3 === 0) cell.classList.add('is-active');
  });

  tick();
  setInterval(tick, 160);
}

/* Copy the clone command */
function initClonePill() {
  const btn = document.querySelector('[data-copy]');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    const text = btn.getAttribute('data-copy');
    try {
      await navigator.clipboard.writeText(text);
      const original = btn.textContent;
      btn.textContent = 'Copied';
      setTimeout(() => (btn.textContent = original), 1600);
    } catch (e) {
      /* clipboard unavailable — no-op */
    }
  });
}
