const form = document.getElementById("predictionForm");
const resultBox = document.getElementById("result");
const loadingBox = document.getElementById("loading");

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    resultBox.classList.add("hidden");
    loadingBox.classList.remove("hidden");

    const customerData = {
        gender: document.getElementById("gender").value,
        SeniorCitizen: Number(document.getElementById("SeniorCitizen").value),
        Partner: document.getElementById("Partner").value,
        Dependents: document.getElementById("Dependents").value,
        tenure: Number(document.getElementById("tenure").value),
        PhoneService: document.getElementById("PhoneService").value,
        MultipleLines: document.getElementById("MultipleLines").value,
        InternetService: document.getElementById("InternetService").value,
        OnlineSecurity: document.getElementById("OnlineSecurity").value,
        OnlineBackup: document.getElementById("OnlineBackup").value,
        DeviceProtection: document.getElementById("DeviceProtection").value,
        TechSupport: document.getElementById("TechSupport").value,
        StreamingTV: document.getElementById("StreamingTV").value,
        StreamingMovies: document.getElementById("StreamingMovies").value,
        Contract: document.getElementById("Contract").value,
        PaperlessBilling: document.getElementById("PaperlessBilling").value,
        PaymentMethod: document.getElementById("PaymentMethod").value,
        MonthlyCharges: Number(document.getElementById("MonthlyCharges").value),
        TotalCharges: Number(document.getElementById("TotalCharges").value)
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(customerData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        const prediction = data.churn_prediction;
        const probability = (data.churn_probability * 100).toFixed(2);

        resultBox.classList.remove("hidden", "churn", "no-churn");

        if (prediction === 1 || prediction === "1" || prediction === true) {
            resultBox.classList.add("churn");
            resultBox.innerHTML = `
                <h2>Customer May Churn</h2>
                <p>Churn Probability: <strong>${probability}%</strong></p>
                <p>Business Action: Contact the customer and offer retention support.</p>
            `;
        } else {
            resultBox.classList.add("no-churn");
            resultBox.innerHTML = `
                <h2>Customer May Stay</h2>
                <p>Churn Probability: <strong>${probability}%</strong></p>
                <p>Business Action: Continue regular customer engagement.</p>
            `;
        }
    } catch (error) {
        resultBox.classList.remove("hidden");
        resultBox.classList.add("churn");
        resultBox.innerHTML = `
            <h2>Error</h2>
            <p>${error.message}</p>
        `;
    } finally {
        loadingBox.classList.add("hidden");
    }
});
