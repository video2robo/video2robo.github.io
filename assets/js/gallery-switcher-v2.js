
// Experiment Switching Logic
function showExperiment(expId) {
    // 1. Hide all experiment content divs
    const contents = document.querySelectorAll('.experiment-content');
    contents.forEach(div => div.style.display = 'none');
    contents.forEach(div => div.classList.remove('active'));

    // 2. Remove active class from all tabs
    const tabs = document.querySelectorAll('.exp-tab');
    tabs.forEach(tab => tab.classList.remove('active'));

    // 3. Show selected experiment
    const selectedDiv = document.getElementById(expId);
    if (selectedDiv) {
        selectedDiv.style.display = 'block';
        selectedDiv.classList.add('active');
    }

    // 4. Set active tab (finding by onclick attribute is messy, better to pass `this` or use event delegation)
    // For simplicity with the inline onclicks: update the tab that targets this expId
    const activeTab = Array.from(tabs).find(t => t.getAttribute('onclick').includes(expId));
    if (activeTab) activeTab.classList.add('active');
}

// Flow Switching Logic
function setFlowType(expId, flowType) {
    const container = document.getElementById(expId);
    if (!container) return;

    // 1. Update buttons
    const buttons = container.querySelectorAll('.flow-btn');
    buttons.forEach(btn => {
        if (btn.getAttribute('onclick').includes(flowType)) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // 2. Toggle images
    const objectFlows = container.querySelectorAll('.object-flow');
    const handFlows = container.querySelectorAll('.hand-flow');

    if (flowType === 'object') {
        objectFlows.forEach(img => img.style.display = 'block');
        handFlows.forEach(img => img.style.display = 'none');
    } else {
        objectFlows.forEach(img => img.style.display = 'none');
        handFlows.forEach(img => img.style.display = 'block');
    }
}

// Global Flow Toggle - applies to all experiments at once
function setGlobalFlowType(flowType) {
    // 1. Apply to all experiment containers
    const experiments = document.querySelectorAll('.experiment-content');
    experiments.forEach(exp => {
        const objectFlows = exp.querySelectorAll('.object-flow');
        const handFlows = exp.querySelectorAll('.hand-flow');

        if (flowType === 'object') {
            objectFlows.forEach(el => el.style.display = 'block');
            handFlows.forEach(el => el.style.display = 'none');
        } else {
            objectFlows.forEach(el => el.style.display = 'none');
            handFlows.forEach(el => el.style.display = 'block');
        }
    });

    // 2. Sync all toggle buttons based on data attribute
    const allBtns = document.querySelectorAll('.flow-header-btn');
    allBtns.forEach(btn => {
        if (btn.getAttribute('data-flow-type') === flowType) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

// Event Delegation for Flow Buttons
// Event Delegation for Flow Buttons
// Event Delegation for Flow Buttons
document.body.addEventListener('click', function (e) {
    if (e.target && e.target.classList.contains('flow-header-btn')) {
        const flowType = e.target.getAttribute('data-flow-type');
        if (flowType) {
            setGlobalFlowType(flowType);
        }
    }
});

// Camera View Switching Logic (Front/Side)
function switchView(videoId, viewType) {
    const video = document.getElementById(videoId);
    if (!video) return;

    // Assume naming convention: 
    // - Current src: assets/mp4/placeholder.mp4
    // - Desired src: assets/mp4/placeholder_side.mp4 (example)
    // But wait, the videoId is specific. We need to know the specific paths.
    // Or pass the path directly?
    // Let's assume we pass the full path or suffix.
    // Actually, better to store paths in data attributes or pass them.
    // BUT the user asked for a general solution.

    // Let's assume conditional toggles where the onclick has the path.
    // E.g. onclick="switchView('vid1', 'assets/mp4/real_side.mp4')"

    // However, my plan said: "switchView(videoId, viewType)"

    // Let's implement robust switching that swaps the source based on data attributes if provided,
    // or simple path replacement.

    // Better approach:
    // The specific step will have:
    // <button onclick="switchView('vid_id', 'front')">Front</button>
    // <button onclick="switchView('vid_id', 'side')">Side</button>

    // And the video element should have `data-front="path/to/front.mp4"` and `data-side="path/to/side.mp4"`.

    const frontPath = video.getAttribute('data-front');
    const sidePath = video.getAttribute('data-side');

    // If data attributes exist, use them.
    if (frontPath && sidePath) {
        const currentTime = video.currentTime;
        const isPaused = video.paused;

        video.src = (viewType === 'side') ? sidePath : frontPath;

        // Attempt to restore state
        video.currentTime = currentTime;
        if (!isPaused) video.play();
    }

    // Toggle UI state
    const wrapper = video.closest('.video-wrapper');
    if (wrapper) {
        const btns = wrapper.querySelectorAll('.view-toggle button');
        btns.forEach(btn => {
            // Simple text check or logic
            if (btn.innerText.toLowerCase().includes(viewType)) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }
}

// Image View Switching Logic (Front/Side) - same pattern as switchView but for <img>
function switchImgView(imgId, viewType) {
    const img = document.getElementById(imgId);
    if (!img) return;

    const frontPath = img.getAttribute('data-front');
    const sidePath = img.getAttribute('data-side');

    if (frontPath && sidePath) {
        img.src = (viewType === 'side') ? sidePath : frontPath;
    }

    // Toggle UI state
    const wrapper = img.closest('.video-wrapper') || img.closest('.asset-col');
    if (wrapper) {
        const btns = wrapper.querySelectorAll('.view-toggle button');
        btns.forEach(btn => {
            if (btn.innerText.toLowerCase().includes(viewType)) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }
}
