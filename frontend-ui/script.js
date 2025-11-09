// IMPORTANT: Replace this with your actual API Gateway deployment endpoint
const API_BASE_URL = "https://YOUR_API_GATEWAY_INVOKE_URL"; 

const triggerBtn = document.getElementById('trigger-run-btn');
const refreshBtn = document.getElementById('refresh-btn');
const fileList = document.getElementById('file-list');
const loader = document.getElementById('loader');
const statusMessage = document.getElementById('status-message');

// All helper functions (showStatus, hideStatus, formatBytes) are identical to previous solutions.

async function fetchPdfList() {
    fileList.innerHTML = '';
    loader.classList.remove('hidden');
    hideStatus();
    try {
        const response = await fetch(`${API_BASE_URL}/reports`); // Updated endpoint
        if (!response.ok) throw new Error(`Failed to fetch file list. Status: ${response.status}`);
        const files = await response.json();
        
        if (files.length === 0) {
            fileList.innerHTML = '<li>No reports found.</li>';
        } else {
            files.forEach(file => {
                const li = document.createElement('li');
                const modifiedDate = new Date(file.last_modified).toLocaleString();
                li.innerHTML = `
                    <div>
                        <a href="#" data-filename="${file.name}">${file.name}</a>
                    </div>
                    <span>${modifiedDate} (${formatBytes(file.size)})</span>
                `;
                fileList.appendChild(li);
            });
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        loader.classList.add('hidden');
    }
}

async function handleDownloadClick(event) {
    if (event.target.tagName !== 'A') return;
    event.preventDefault();
    const filename = event.target.dataset.filename;
    showStatus(`Preparing download for ${filename}...`, 'info');
    
    try {
        const response = await fetch(`${API_BASE_URL}/reports/${filename}`); // Updated endpoint
        if (!response.ok) throw new Error(`Failed to get download URL. Status: ${response.status}`);
        
        const data = await response.json();
        if (data.download_url) {
            window.location.href = data.download_url;
            hideStatus();
        } else {
            throw new Error("Download URL not found in response.");
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
    }
}

async function triggerPipelineRun() {
    triggerBtn.disabled = true;
    triggerBtn.textContent = 'Generating...';
    showStatus('Pipeline triggered. This may take several minutes. The list will not refresh automatically.', 'info');

    try {
        // Here we can eventually add support for passing tickers/dates from the UI
        const body = {}; 
        
        const response = await fetch(`${API_BASE_URL}/trigger`, { 
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
        });
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.message || `Failed to trigger pipeline. Status: ${response.status}`);
        }
        const result = await response.json();
        showStatus(`Backend response: ${result.message}`, 'success');
        
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        triggerBtn.disabled = false;
        triggerBtn.textContent = 'Manually Generate New Report';
    }
}

// Event Listeners (identical to previous solutions)
document.addEventListener('DOMContentLoaded', fetchPdfList);
refreshBtn.addEventListener('click', fetchPdfList);
triggerBtn.addEventListener('click', triggerPipelineRun);
fileList.addEventListener('click', handleDownloadClick);