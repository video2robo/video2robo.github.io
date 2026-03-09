// Gallery navigation
document.addEventListener('DOMContentLoaded', function () {
  const galleries = [
    {
      sectionId: 'gallery-section-cup_on_plate',
      galleryInnerId: 'cup_on_plate',
      scrollLeftBtnId: 'scrollLeftBtncup_on_plate',
      scrollRightBtnId: 'scrollRightBtncup_on_plate'
    },
    {
      sectionId: 'gallery-section-open_drawer',
      galleryInnerId: 'open_drawer',
      scrollLeftBtnId: 'scrollLeftBtnopen_drawer',
      scrollRightBtnId: 'scrollRightBtnopen_drawer'
    },
    {
      sectionId: 'gallery-section-hang_mug',
      galleryInnerId: 'hang_mug',
      scrollLeftBtnId: 'scrollLeftBtnhang_mug',
      scrollRightBtnId: 'scrollRightBtnhang_mug'
    },
    {
      sectionId: 'gallery-section-block_insertion',
      galleryInnerId: 'block_insertion',
      scrollLeftBtnId: 'scrollLeftBtnblock_insertion',
      scrollRightBtnId: 'scrollRightBtnblock_insertion'
    },
    {
      sectionId: 'gallery-section-water_plant',
      galleryInnerId: 'water_plant',
      scrollLeftBtnId: 'scrollLeftBtnwater_plant',
      scrollRightBtnId: 'scrollRightBtnwater_plant'
    },
    {
      sectionId: 'gallery-section-open_lid',
      galleryInnerId: 'open_lid',
      scrollLeftBtnId: 'scrollLeftBtnopen_lid',
      scrollRightBtnId: 'scrollRightBtnopen_lid'
    },
    {
      sectionId: 'gallery-section-gait_transition',
      galleryInnerId: 'gait_transition',
      scrollLeftBtnId: 'scrollLeftBtnGait_transition',
      scrollRightBtnId: 'scrollRightBtnGait_transition'
    },
    {
      sectionId: 'gallery-section-stability',
      galleryInnerId: 'stability',
      scrollLeftBtnId: 'scrollLeftBtnStability',
      scrollRightBtnId: 'scrollRightBtnStability'
    },
    {
      sectionId: 'gallery-section-failures',
      galleryInnerId: 'failures',
      scrollLeftBtnId: 'scrollLeftBtnFailures',
      scrollRightBtnId: 'scrollRightBtnFailures'
    },
    {
      sectionId: 'gallery-section-failure_mode_1',
      galleryInnerId: 'failure_mode_1',
      scrollLeftBtnId: 'scrollLeftBtnfailure_mode_1',
      scrollRightBtnId: 'scrollRightBtnfailure_mode_1'
    },
    {
      sectionId: 'gallery-section-failure',
      galleryInnerId: 'failure',
      scrollLeftBtnId: 'scrollLeftBtnfailure',
      scrollRightBtnId: 'scrollRightBtnfailure'
    }
  ];

  galleries.forEach(galleryConfig => {
    const gallerySection = document.getElementById(galleryConfig.sectionId);
    if (!gallerySection) {
      console.error(`Gallery section with ID ${galleryConfig.sectionId} not found.`);
      return;
    }

    const galleryContainer = gallerySection.querySelector('.video-gallery-container');
    const galleryInner = document.getElementById(galleryConfig.galleryInnerId);
    const scrollLeftBtn = document.getElementById(galleryConfig.scrollLeftBtnId);
    const scrollRightBtn = document.getElementById(galleryConfig.scrollRightBtnId);

    if (galleryContainer && galleryInner && scrollLeftBtn && scrollRightBtn) {
      // Calculate the scroll amount based on the width of the first video + gap
      const scrollAmount = (galleryInner.firstElementChild?.offsetWidth || 300) + 15; // 15 is the gap

      scrollLeftBtn.addEventListener('click', () => {
        // Scroll the CONTAINER element
        galleryContainer.scrollBy({ left: -scrollAmount, behavior: 'smooth' });
      });

      scrollRightBtn.addEventListener('click', () => {
        // Scroll the CONTAINER element
        galleryContainer.scrollBy({ left: scrollAmount, behavior: 'smooth' });
      });
    } else {
      console.error(`Gallery elements not found for navigation setup in section ${galleryConfig.sectionId}.`);
      // Log which elements might be missing
      if (!galleryContainer) console.error('Missing element: .video-gallery-container in section ' + galleryConfig.sectionId);
      if (!galleryInner) console.error(`Missing element with ID ${galleryConfig.galleryInnerId}`);
      if (!scrollLeftBtn) console.error(`Missing element with ID ${galleryConfig.scrollLeftBtnId}`);
      if (!scrollRightBtn) console.error(`Missing element with ID ${galleryConfig.scrollRightBtnId}`);
    }
  });
});
