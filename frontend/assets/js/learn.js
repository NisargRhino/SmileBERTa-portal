/* Learn pages: build the table of contents and highlight the active section. */
(function () {
  const { $, $$ } = SB;
  const toc = $('.doc-toc'); const heads = $$('.doc h2[id]');
  if (toc && heads.length) {
    toc.insertAdjacentHTML('beforeend', heads.map((h) => `<a href="#${h.id}">${h.textContent}</a>`).join(''));
    const links = $$('.doc-toc a[href^="#"]');
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) links.forEach((l) => l.classList.toggle('active', l.getAttribute('href') === '#' + e.target.id));
    }), { rootMargin: '-20% 0px -70% 0px' });
    heads.forEach((h) => io.observe(h));
  }
  // Animate figures when they scroll into view.
  const fio = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); fio.unobserve(e.target); } }), { threshold: 0.25 });
  $$('.figure').forEach((f) => fio.observe(f));
})();
