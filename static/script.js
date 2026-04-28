let lastData = null;

async function analyze() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload an image");
        return;
    }

    // Preview
    document.getElementById("preview").innerHTML =
        `<img src="${URL.createObjectURL(file)}" width="200">`;

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("results").classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/analyze", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    lastData = data;

    document.getElementById("loading").classList.add("hidden");
    document.getElementById("results").classList.remove("hidden");

    // Predictions
    const predDiv = document.getElementById("predictions");
    predDiv.innerHTML = "";

    data.predictions.forEach(p => {
        predDiv.innerHTML += `
            <div class="card">
                ${p.label} - ${(p.score * 100).toFixed(2)}%
            </div>
        `;
    });

    // Color
    const colorBox = document.getElementById("colorBox");
    colorBox.style.background = data.color;
    colorBox.innerText = data.color;
}

async function askAI() {
    const question = document.getElementById("question").value;

    if (!question || !lastData) {
        alert("Analyze an image first and enter a question");
        return;
    }

    const res = await fetch("/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            question: question,
            predictions: lastData.predictions,
            color: lastData.color
        })
    });

    const data = await res.json();

    document.getElementById("aiAnswer").innerText = data.answer;
}