let lastData = null;

async function analyze() {
    const file = document.getElementById("fileInput").files[0];
    if (!file) return alert("Upload image");

    document.getElementById("preview").innerHTML =
        `<img src="${URL.createObjectURL(file)}" class="preview">`;

    document.getElementById("loadingOverlay").classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/analyze", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

   if (data.error) {
    document.getElementById("loadingOverlay").classList.add("hidden");
    alert(data.error);
    return;
}

    lastData = data;

    document.getElementById("loading").classList.add("hidden");
    document.getElementById("results").classList.remove("hidden");

    let html = "";
    data.predictions.forEach(p => {
        html += `<div class="pill">${p.label} (${(p.score*100).toFixed(1)}%)</div>`;
    });

    document.getElementById("predictions").innerHTML = html;
    document.getElementById("caption").innerText = data.caption;

    const colorBox = document.getElementById("colorBox");
    colorBox.style.background = data.color;
    colorBox.innerText = data.color;

    document.getElementById("comparison").innerText = data.comparison;
}

async function askAI() {
    const q = document.getElementById("question").value;

    const res = await fetch("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...lastData, question: q})
    });

    const data = await res.json();

    document.getElementById("chat").innerHTML +=
        `<div class="chat"><b>You:</b> ${q}<br><b>AI:</b> ${data.answer}</div>`;
}