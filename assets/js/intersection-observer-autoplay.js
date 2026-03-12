// intersection-observer-autoplay.js
document.addEventListener('DOMContentLoaded', function () {
  // 只选取你想要懒加载的那一批视频
  const videos = document.querySelectorAll('.step-row video');

  // Intersection Observer 配置
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const video = entry.target;
      if (entry.isIntersecting) {
        // 进入视口，自动播放
        video.play();
      } else {
        // 离开视口，暂停
        video.pause();
      }
    });
  }, {
    threshold: 0.3 // 30%可见时触发
  });

  videos.forEach(video => {
    observer.observe(video);
  });
});