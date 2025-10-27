// Lauschomat Web UI

// State
let currentDate = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
let currentPage = 1;
let pageSize = 20;
let totalTransmissions = 0;
let selectedTransmissionId = null;
let wavesurfer = null;

// DOM Elements
const dateSelect = document.getElementById('date-select');
const transmissionsList = document.getElementById('transmissions-list');
const prevPageBtn = document.getElementById('prev-page');
const nextPageBtn = document.getElementById('next-page');
const pageInfo = document.getElementById('page-info');
const detailContent = document.querySelector('.detail-content');
const noSelection = document.querySelector('.no-selection');

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initWaveSurfer();
    loadDates();
    
    // Event listeners
    dateSelect.addEventListener('change', () => {
        currentDate = dateSelect.value;
        currentPage = 1;
        loadTransmissions();
    });
    
    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            loadTransmissions();
        }
    });
    
    nextPageBtn.addEventListener('click', () => {
        const totalPages = Math.ceil(totalTransmissions / pageSize);
        if (currentPage < totalPages) {
            currentPage++;
            loadTransmissions();
        }
    });
    
    document.getElementById('play-btn').addEventListener('click', () => {
        if (wavesurfer) {
            wavesurfer.play();
        }
    });
    
    document.getElementById('stop-btn').addEventListener('click', () => {
        if (wavesurfer) {
            wavesurfer.stop();
        }
    });
});

// Initialize WaveSurfer
function initWaveSurfer() {
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#3498db',
        progressColor: '#2980b9',
        height: 80,
        responsive: true,
        cursorWidth: 2,
        cursorColor: '#e74c3c',
        barWidth: 2,
        barGap: 1
    });
    
    wavesurfer.on('ready', () => {
        console.log('WaveSurfer ready');
    });
    
    wavesurfer.on('error', (err) => {
        console.error('WaveSurfer error:', err);
    });
}

// Load available dates
async function loadDates() {
    try {
        const response = await fetch('/api/dates');
        const data = await response.json();
        
        // Clear select
        dateSelect.innerHTML = '';
        
        // Add options
        data.dates.forEach(date => {
            const option = document.createElement('option');
            option.value = date;
            option.textContent = formatDate(date);
            dateSelect.appendChild(option);
        });
        
        // Set current date if available, otherwise use first date
        if (data.dates.includes(currentDate)) {
            dateSelect.value = currentDate;
        } else if (data.dates.length > 0) {
            currentDate = data.dates[0];
            dateSelect.value = currentDate;
        }
        
        // Load transmissions for selected date
        loadTransmissions();
    } catch (error) {
        console.error('Error loading dates:', error);
        transmissionsList.innerHTML = '<div class="error">Failed to load dates</div>';
    }
}

// Load transmissions for the current date and page
async function loadTransmissions() {
    try {
        transmissionsList.innerHTML = '<div class="loading">Loading...</div>';
        
        const offset = (currentPage - 1) * pageSize;
        const response = await fetch(`/api/transmissions?date=${currentDate}&limit=${pageSize}&offset=${offset}`);
        const data = await response.json();
        
        totalTransmissions = data.total;
        updatePagination();
        
        if (data.transmissions.length === 0) {
            transmissionsList.innerHTML = '<div class="no-data">No transmissions found for this date</div>';
            return;
        }
        
        // Render transmissions
        transmissionsList.innerHTML = '';
        data.transmissions.forEach(transmission => {
            const item = document.createElement('div');
            item.className = 'transmission-item';
            if (transmission.id === selectedTransmissionId) {
                item.classList.add('selected');
            }
            
            const time = new Date(transmission.timestamp_utc).toLocaleTimeString();
            const duration = formatDuration(transmission.duration_sec);
            
            item.innerHTML = `
                <div class="transmission-time">${time}</div>
                <div class="transmission-duration">${duration}</div>
                <div class="transmission-text">${transmission.text || 'No transcription available'}</div>
            `;
            
            item.addEventListener('click', () => {
                selectTransmission(transmission.id);
            });
            
            transmissionsList.appendChild(item);
        });
    } catch (error) {
        console.error('Error loading transmissions:', error);
        transmissionsList.innerHTML = '<div class="error">Failed to load transmissions</div>';
    }
}

// Update pagination controls
function updatePagination() {
    const totalPages = Math.ceil(totalTransmissions / pageSize);
    pageInfo.textContent = `Page ${currentPage} of ${totalPages || 1}`;
    
    prevPageBtn.disabled = currentPage <= 1;
    nextPageBtn.disabled = currentPage >= totalPages;
}

// Select a transmission and load its details
async function selectTransmission(id) {
    try {
        selectedTransmissionId = id;
        
        // Update selection in list
        const items = transmissionsList.querySelectorAll('.transmission-item');
        items.forEach(item => item.classList.remove('selected'));
        const selectedItem = Array.from(items).find(item => 
            item.querySelector('.transmission-text').textContent.includes(id));
        if (selectedItem) {
            selectedItem.classList.add('selected');
        }
        
        // Show loading state
        detailContent.style.display = 'block';
        noSelection.style.display = 'none';
        document.getElementById('transcription-text').innerHTML = '<div class="loading">Loading transcription...</div>';
        
        // Fetch transmission details
        const response = await fetch(`/api/transmission/${id}`);
        const transmission = await response.json();
        
        // Update metadata
        document.getElementById('detail-id').textContent = transmission.id;
        document.getElementById('detail-time').textContent = new Date(transmission.timestamp_utc).toLocaleString();
        document.getElementById('detail-duration').textContent = formatDuration(transmission.duration_sec);
        document.getElementById('detail-device').textContent = transmission.device;
        
        // Load audio
        if (transmission.audio_path) {
            wavesurfer.load(`/media/${transmission.audio_path}`);
        }
        
        // Update transcription
        if (transmission.transcription) {
            document.getElementById('transcription-text').textContent = transmission.transcription.text || 'No text available';
            document.getElementById('transcription-model').textContent = transmission.transcription.model || 'Unknown';
            document.getElementById('transcription-confidence').textContent = 
                transmission.transcription.confidence ? 
                `${Math.round(transmission.transcription.confidence * 100)}%` : 
                'N/A';
        } else {
            document.getElementById('transcription-text').textContent = transmission.text || 'No transcription available';
            document.getElementById('transcription-model').textContent = transmission.model || 'Unknown';
            document.getElementById('transcription-confidence').textContent = 
                transmission.confidence ? 
                `${Math.round(transmission.confidence * 100)}%` : 
                'N/A';
        }
    } catch (error) {
        console.error('Error loading transmission details:', error);
        document.getElementById('transcription-text').innerHTML = '<div class="error">Failed to load transcription</div>';
    }
}

// Format date for display
function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString(undefined, { 
        weekday: 'short', 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    });
}

// Format duration for display
function formatDuration(seconds) {
    if (!seconds) return '0:00';
    
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}
