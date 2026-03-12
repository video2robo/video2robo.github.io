document.addEventListener('DOMContentLoaded', function () {
  const taskButtonsContainer = document.getElementById('task-buttons');
  const taskViewer = document.getElementById('task-viewer');
  // const initialObservationImg = document.getElementById('initial-observation');
  const inputVideo = document.getElementById('input-video');
  const poseVideo = document.getElementById('pose-video');
  // const executionVideo = document.getElementById('execution-video');
  const trajectorySelect = document.getElementById('trajectory-type');

  // Per-step configuration
  // Viser viewers are self-contained in assets/viser-client/assets/fmb_fake_viser/stepN/
  const viserBase = 'assets/viser-client/assets/fmb_fake_viser';
  const stepConfig = {
    Attach: {
      assetsDir: 'assets/mp4/attach',
      inputVideo: '0__attach.mp4',
      poseVideo: 'obj_all.mp4',
      viserObject: `${viserBase}/Attach/viewer/index.html`,
      viserDemo: `${viserBase}/Attach/viewer_demo/index.html`,
    },
    Drum: {
      assetsDir: 'assets/mp4/drum',
      inputVideo: '0__drum.mp4',
      poseVideo: 'obj_all.mp4',
        viserObject: `${viserBase}/Drum/viewer/index.html`,
        viserDemo: `${viserBase}/Drum/viewer_demo/index.html`,
    },
    Place: {
      assetsDir: 'assets/mp4/place',
      inputVideo: '0__place.mp4',
      poseVideo: 'obj_all.mp4',
      viserObject: `${viserBase}/Place/viewer/index.html`,
      viserDemo: `${viserBase}/Place/viewer_demo/index.html`,
    },
    Pour: {
      assetsDir: 'assets/mp4/pour',
      inputVideo: '0__pour.mp4',
      poseVideo: 'obj_all.mp4',
      viserObject: `${viserBase}/Pour/viewer/index.html`,
      viserDemo: `${viserBase}/Pour/viewer_demo/index.html`,
    },
    Stack: {
      assetsDir: 'assets/mp4/stack',
      inputVideo: '0__stack.mp4',
      poseVideo: 'obj_all.mp4',
      viserObject: `${viserBase}/Stack/viewer/index.html`,
      viserDemo: `${viserBase}/Stack/viewer_demo/index.html`,
    },
    Sweep: {
      assetsDir: 'assets/mp4/sweep',
      inputVideo: '0__sweep.mp4',
      poseVideo: 'obj_all.mp4',
      viserObject: `${viserBase}/Sweep/viewer/index.html`,
      viserDemo: `${viserBase}/Sweep/viewer_demo/index.html`,
    },
  };

  // Build viser iframe URL for a given step + trajectory type
  const buildSrc = (step, trajectory) => {
    const config = stepConfig[step];
    if (!config) return '';
    // Each step has its own self-contained viser viewer
    return trajectory === 'demo' ? config.viserDemo : config.viserObject;
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
      trajectorySelect.value = 'demo';
    } else {
      if (objectFlowOption) objectFlowOption.style.display = '';
    }

    const trajectory = trajectorySelect.value;

    // Update media sources
    // initialObservationImg.src = `${config.assetsDir}/${config.startImg}`;
    inputVideo.src = `${config.assetsDir}/${config.inputVideo}`;
    poseVideo.src = `${config.assetsDir}/${config.poseVideo}`;
    // executionVideo.src = `${config.assetsDir}/${config.execVideo}`;

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
      flowLabel.textContent = trajectorySelect.value === 'demo'
        ? 'Clean Robot Demo'
        : 'Object 6D Pose';
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
      flowLabel.textContent = trajectory === 'demo'
        ? 'Clean Robot Demo'
        : 'Object 6D Pose';
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
  // const video = document.getElementById('execution-video');
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
