document.addEventListener("DOMContentLoaded", function () {
    fetchECGData();
});

function fetchECGData() {
    fetch("http://127.0.0.1:5000/ecg-data") // Update if hosted elsewhere
        .then(response => response.json())
        .then(data => displayECGData(data))
        .catch(error => console.error("Error fetching ECG data:", error));
}

function displayECGData(ecgData) {
    const container = document.getElementById("ecg-report-container");
    container.innerHTML = ""; // Clear existing content

    if (ecgData.length === 0) {
        container.innerHTML = "<p>No ECG records available.</p>";
        return;
    }

    ecgData.forEach(record => {
        const ecgBox = document.createElement("div");
        ecgBox.classList.add("service-box");

        ecgBox.innerHTML = `
            <div class="service-info">
                <h4>Patient ID: ${record.patient_id}</h4>
                <p><strong>Heart Rate:</strong> ${record.heart_rate} BPM</p>
                <p><strong>Status:</strong> ${record.status}</p>
                <p><strong>Risk Level:</strong> ${record.risk_level}</p>
            </div>
        `;

        container.appendChild(ecgBox);
    });
}
