// Hero video initialization
function startHero() {
  const vid = document.getElementById('bg-video');
  if (!vid) return;
  vid.play().catch(() => {});
  vid.style.opacity = 1;
}

// Smooth scroll to content
function scrollToContent() {
  const content = document.querySelector('.main-content');
  if (content) {
    content.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
  }
}

// Auto-start hero on page load
document.addEventListener('DOMContentLoaded', function () {
  startHero();
});