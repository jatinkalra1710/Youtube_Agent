document.getElementById("askBtn").addEventListener("click", async () => {
    let [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });

    const url = tab.url;
    const question = document.getElementById("question").value;

    const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            url,
            question,
            language: "English"
        })
    });

    const data = await res.json();

    document.getElementById("answer").innerText = data.answer;
});
