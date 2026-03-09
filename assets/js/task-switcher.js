document.addEventListener('DOMContentLoaded', function () {
  const taskButtonsContainer = document.getElementById('task-buttons');
  const taskViewer = document.getElementById('task-viewer');
  const initialObservationImg = document.getElementById('initial-observation');
  const generatedVideo = document.getElementById('generated-video');
  const executionVideo = document.getElementById('execution-video');
  const trajectorySelect = document.getElementById('trajectory-type');

  // Per-step configuration
  // Viser viewers are self-contained in assets/viser-client/assets/fmb_fake_viser/stepN/
  const viserBase = 'assets/viser-client/assets/fmb_fake_viser';
  const stepConfig = {
    step_1: {
      assetsDir: 'assets/mp4/fmb_fake/step_1',
      startImg: 'start.png',
      genVideo: 'rgb_video_16fps_1.mp4',
      execVideo: 'camera0_1.mp4',
      viserObject: `${viserBase}/step1/viewer/index.html`,
      viserHand: `${viserBase}/step1/viewer_hand/index.html`,
    },
    step_2: {
      assetsDir: 'assets/mp4/fmb_fake/step_2',
      startImg: 'start.png',
      genVideo: 'rgb_video_16fps_2.mp4',
      execVideo: 'camera0_2.mp4',
      viserObject: `${viserBase}/step2/viewer/index.html`,
      viserHand: `${viserBase}/step2/viewer_hand/index.html`,
    },
    step_3: {
      assetsDir: 'assets/mp4/fmb_fake/step_3',
      startImg: 'start.png',
      genVideo: 'rgb_video_16fps_3.mp4',
      execVideo: 'camera0_3.mp4',
      viserObject: `${viserBase}/step3/viewer/index.html`,
      viserHand: `${viserBase}/step3/viewer_hand/index.html`,
    },
    step_4: {
      assetsDir: 'assets/mp4/fmb_fake/step_4',
      startImg: 'start.png',
      genVideo: 'rgb_video_16fps_4.mp4',
      execVideo: 'camera0_4.mp4',
      viserObject: `${viserBase}/step4/viewer/index.html`,
      viserHand: `${viserBase}/step4/viewer_hand/index.html`,
    },
    step_4_recovery: {
      assetsDir: 'assets/mp4/fmb_fake/step_4_recovery',
      startImg: 'start_clean.png',
      genVideo: 'rgb_video_16fps_4_recovery.mp4',
      execVideo: 'execution_front_view.mp4',
      execVideoSide: 'execution_side_view.mp4',
      // step_4_recovery only has hand flow, no object flow
      viserObject: `${viserBase}/step4_recovery/viewer_hand/index.html`,
      viserHand: `${viserBase}/step4_recovery/viewer_hand/index.html`,
    },
  };

  // Build viser iframe URL for a given step + trajectory type
  const buildSrc = (step, trajectory) => {
    const config = stepConfig[step];
    if (!config) return '';
    // Each step has its own self-contained viser viewer
    return trajectory === 'hand' ? config.viserHand : config.viserObject;
  };

  // Currently active step
  let currentStep = 'step_1';

  const updateMediaSources = (step) => {
    currentStep = step;
    const config = stepConfig[step];
    if (!config) return;

    // Handle flow options dropdown for step_4_recovery (no object flow)
    const objectFlowOption = trajectorySelect.querySelector('option[value="object"]');
    if (step === 'step_4_recovery') {
      if (objectFlowOption) objectFlowOption.style.display = 'none';
      trajectorySelect.value = 'hand';
    } else {
      if (objectFlowOption) objectFlowOption.style.display = '';
    }

    const trajectory = trajectorySelect.value;

    // Update media sources
    initialObservationImg.src = `${config.assetsDir}/${config.startImg}`;
    generatedVideo.src = `${config.assetsDir}/${config.genVideo}`;
    executionVideo.src = `${config.assetsDir}/${config.execVideo}`;

    // Update viser iframe
    taskViewer.src = buildSrc(step, trajectory);
    // Show/Hide View Toggle for Step 4 Recovery
    const toggleContainer = document.getElementById('interactive-view-toggle');
    if (toggleContainer) {
      if (step === 'step_4_recovery') {
        toggleContainer.style.display = 'flex';
        // Reset to front view by default when switching steps
        switchInteractiveView('front');
      } else {
        toggleContainer.style.display = 'none';
      }
    }

    // Update the caption label
    const flowLabel = document.getElementById('flow-type-label');
    if (flowLabel) {
      flowLabel.textContent = trajectorySelect.value === 'hand'
        ? 'Actionable Hand Flow'
        : 'Actionable Object Flow';
    }
  };

  // Step tab clicks
  taskButtonsContainer.addEventListener('click', function (e) {
    if (e.target.classList.contains('task-button')) {
      taskButtonsContainer.querySelectorAll('.task-button').forEach(button => {
        button.classList.remove('active');
      });
      e.target.classList.add('active');
      const selectedTask = e.target.dataset.value;
      updateMediaSources(selectedTask);
    }
  });

  // Trajectory dropdown changes — only reload the iframe
  trajectorySelect.addEventListener('change', function () {
    const trajectory = trajectorySelect.value;
    taskViewer.src = buildSrc(currentStep, trajectory);

    // Update the caption label
    const flowLabel = document.getElementById('flow-type-label');
    if (flowLabel) {
      flowLabel.textContent = trajectory === 'hand'
        ? 'Actionable Hand Flow'
        : 'Actionable Object Flow';
    }
  });

  // Set the initial view
  const initialButton = taskButtonsContainer.querySelector('.task-button.active');
  if (initialButton) {
    const initialTask = initialButton.dataset.value;
    updateMediaSources(initialTask);
  }
});

// Interactive Viewer Switch Function (Global)
window.switchInteractiveView = function (viewType) {
  const video = document.getElementById('execution-video');
  const toggleContainer = document.getElementById('interactive-view-toggle');

  if (!video || !toggleContainer) return;

  const frontPath = 'assets/mp4/fmb_fake/step_4_recovery/execution_front_view.mp4';
  const sidePath = 'assets/mp4/fmb_fake/step_4_recovery/execution_side_view.mp4';

  // Update Buttons
  const buttons = toggleContainer.querySelectorAll('button');
  buttons.forEach(btn => {
    if (btn.innerText.toLowerCase().includes(viewType)) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  const targetSrc = (viewType === 'side') ? sidePath : frontPath;

  if (!video.src.includes(targetSrc)) {
    const currentTime = video.currentTime;
    const isPaused = video.paused;
    video.src = targetSrc;
    video.currentTime = currentTime;
    if (!isPaused) video.play();
  }
};
