// State management
let currentMode = 'sample';
let selectedFile = null;
let sampleItems = [];

document.addEventListener('DOMContentLoaded', () => {
    loadSampleItems();
    setupDropzone();
});

// TAB SWITCHING
function switchTab(mode) {
    currentMode = mode;
    const tabSampleBtn = document.getElementById('tabSampleBtn');
    const tabUploadBtn = document.getElementById('tabUploadBtn');
    const tabSampleContent = document.getElementById('tabSampleContent');
    const tabUploadContent = document.getElementById('tabUploadContent');

    if (mode === 'sample') {
        tabSampleBtn.classList.add('active');
        tabUploadBtn.classList.remove('active');
        tabSampleContent.classList.remove('hidden');
        tabUploadContent.classList.add('hidden');
    } else {
        tabSampleBtn.classList.remove('active');
        tabUploadBtn.classList.add('active');
        tabSampleContent.classList.add('hidden');
        tabUploadContent.classList.remove('hidden');
    }
}

// SETUP DROPZONE
function setupDropzone() {
    const dropzone = document.getElementById('dropZone');
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--primary)';
        }, false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--border-color)';
        }, false);
    });
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

function handleFileSelect(event) {
    if (event.target.files.length > 0) {
        handleFile(event.target.files[0]);
    }
}

function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImg = document.getElementById('uploadPreviewImg');
        const dropContent = document.getElementById('dropzoneContent');
        previewImg.src = e.target.result;
        previewImg.classList.remove('hidden');
        dropContent.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

// LOAD SAMPLE ITEMS
async function loadSampleItems() {
    try {
        const response = await fetch('/api/sample_items');
        const data = await response.json();
        sampleItems = data.items || [];

        const select = document.getElementById('sampleSelect');
        select.innerHTML = '<option value="">-- Chọn sản phẩm thời trang trong kho --</option>';

        const chipsContainer = document.getElementById('quickSampleChips');
        chipsContainer.innerHTML = '';

        sampleItems.forEach((item, index) => {
            const opt = document.createElement('option');
            opt.value = item.index;
            opt.textContent = `[${item.subCategory} | ${item.gender}] ${item.name} (${item.usage})`;
            select.appendChild(opt);

            // Quick chips for top 6 items
            if (index < 6) {
                const chip = document.createElement('button');
                chip.className = 'chip-btn';
                chip.textContent = `${item.subCategory}: ${item.name.substring(0, 25)}...`;
                chip.onclick = () => {
                    select.value = item.index;
                    document.querySelectorAll('.chip-btn').forEach(c => c.classList.remove('active'));
                    chip.classList.add('active');
                };
                chipsContainer.appendChild(chip);
            }
        });

        if (sampleItems.length > 0) {
            select.selectedIndex = 1;
        }
    } catch (err) {
        console.error("Lỗi khi tải mẫu sản phẩm:", err);
    }
}

// EXECUTE RECOMMENDATION
async function executeRecommendation() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultsSection = document.getElementById('resultsSection');

    const gender = document.getElementById('filterGender').value;
    const usage = document.getElementById('filterUsage').value;
    const model = document.getElementById('filterModel').value;

    loadingOverlay.classList.remove('hidden');

    try {
        let response;
        if (currentMode === 'sample') {
            const sampleIdx = document.getElementById('sampleSelect').value;
            if (sampleIdx === "" || sampleIdx === null) {
                alert("Vui lòng chọn một sản phẩm từ danh mục mẫu!");
                loadingOverlay.classList.add('hidden');
                return;
            }

            response = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query_idx: parseInt(sampleIdx),
                    use_baseline: model === 'baseline',
                    preferred_gender: gender,
                    preferred_usage: usage
                })
            });
        } else {
            if (!selectedFile) {
                alert("Vui lòng tải lên một ảnh trang phục!");
                loadingOverlay.classList.add('hidden');
                return;
            }

            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('use_baseline', model === 'baseline');
            formData.append('preferred_gender', gender);
            formData.append('preferred_usage', usage);

            response = await fetch('/api/recommend', {
                method: 'POST',
                body: formData
            });
        }

        const result = await response.json();
        if (result.error) {
            alert("Lỗi từ server: " + result.error);
        } else {
            renderResults(result, model);
            resultsSection.classList.remove('hidden');
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    } catch (err) {
        alert("Có lỗi xảy ra trong quá trình kết nối API: " + err.message);
    } finally {
        loadingOverlay.classList.add('hidden');
    }
}

// RENDER RESULTS
function renderResults(data, modelChoice) {
    const origin = data.origin;
    const recs = data.recommendations;

    // 1. Render Query Item
    const queryImg = document.getElementById('queryImg');
    const queryName = document.getElementById('queryName');
    const queryTags = document.getElementById('queryTags');

    if (origin.image_url) {
        queryImg.src = origin.image_url;
    } else if (selectedFile) {
        queryImg.src = URL.createObjectURL(selectedFile);
    }

    queryName.textContent = origin.name;

    queryTags.innerHTML = `
        <span class="tag highlight">${origin.subCategory}</span>
        <span class="tag">${origin.gender}</span>
        <span class="tag">${origin.usage}</span>
    `;

    // 2. Badge model
    const badge = document.getElementById('usedModelBadge');
    if (modelChoice === 'baseline') {
        badge.textContent = "HSV Baseline 512D";
        badge.style.color = "#F59E0B";
        badge.style.borderColor = "rgba(245, 158, 11, 0.3)";
        badge.style.background = "rgba(245, 158, 11, 0.15)";
    } else {
        badge.textContent = "ResNet50 Deep Embeddings 2048D";
        badge.style.color = "#10B981";
        badge.style.borderColor = "rgba(16, 185, 129, 0.3)";
        badge.style.background = "rgba(16, 185, 129, 0.15)";
    }

    // 3. Render Outfit Grid
    const outfitGrid = document.getElementById('outfitGrid');
    outfitGrid.innerHTML = '';

    const categories = Object.keys(recs);
    if (categories.length === 0) {
        outfitGrid.innerHTML = '<div style="color:var(--text-muted); grid-column:1/-1;">Không tìm thấy món phối phù hợp với bộ lọc đã chọn.</div>';
        return;
    }

    categories.forEach(cat => {
        const items = recs[cat];
        items.forEach(item => {
            const card = document.createElement('div');
            card.className = 'outfit-card';

            const scorePercent = Math.round(item.score * 100);

            card.innerHTML = `
                <div class="outfit-cat-badge">✦ ${item.subCategory}</div>
                <div class="outfit-img-wrapper">
                    <img src="${item.image_url || '/images/' + item.id + '.jpg'}" alt="${item.name}" class="outfit-img" onerror="this.src='/static/images/placeholder.png'">
                </div>
                <div class="outfit-name" title="${item.name}">${item.name}</div>
                <div class="item-tags" style="margin-bottom: 8px;">
                    <span class="tag">${item.baseColour || 'Đa sắc'}</span>
                    <span class="tag">${item.usage || 'Đa dụng'}</span>
                </div>
                <div class="score-bar-container">
                    <div class="score-meta">
                        <span>Độ tương thích</span>
                        <span class="score-val">${scorePercent}%</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-bar-fill" style="width: ${scorePercent}%;"></div>
                    </div>
                </div>
            `;
            outfitGrid.appendChild(card);
        });
    });
}
