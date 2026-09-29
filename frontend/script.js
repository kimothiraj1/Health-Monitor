const API_BASE = ""; // same-origin: FastAPI serves this page itself.

const form = document.getElementById("wellness-form");
const submitBtn = document.getElementById("submit-btn");
const formError = document.getElementById("form-error");

const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const resultStatus = document.getElementById("result-status");
const resultScore = document.getElementById("result-score");
const resultMessage = document.getElementById("result-message");

const chartCanvas = document.getElementById("history-chart");
const chartEmpty = document.getElementById("chart-empty");

let historyChart = null;

const STATUS_CLASS = {
  "Good": "good",
  "Moderate": "moderate",
  "Needs Improvement": "poor",
};

function showFormError(message) {
  formError.textContent = message;
  formError.hidden = !message;
}

function collectFormData() {
  const fields = [
    "age", "sleep_hours", "steps", "exercise_minutes", "water_intake",
    "stress_level", "heart_rate", "calories", "screen_time", "weight", "mood",
  ];
  const data = {};
  for (const field of fields) {
    const el = document.getElementById(field);
    const value = parseFloat(el.value);
    if (Number.isNaN(value)) {
      throw new Error(`Please enter a value for "${field.replace("_", " ")}".`);
    }
    data[field] = ["age", "steps", "exercise_minutes", "stress_level", "heart_rate", "calories", "mood"].includes(field)
      ? Math.round(value)
      : value;
  }
  return data;
}

function renderResult(result) {
  resultEmpty.hidden = true;
  resultContent.hidden = false;

  resultStatus.textContent = result.status;
  resultStatus.className = "result-status " + (STATUS_CLASS[result.status] || "moderate");
  resultScore.textContent = Math.round(result.wellness_score);
  resultMessage.textContent = result.message;
}

function renderChart(history) {
  if (!history.length) {
    chartCanvas.parentElement.hidden = true;
    chartEmpty.hidden = false;
    return;
  }
  chartCanvas.parentElement.hidden = false;
  chartEmpty.hidden = true;

  const labels = history.map((r) =>
    new Date(r.date).toLocaleDateString(undefined, { month: "short", day: "numeric" })
  );
  const scores = history.map((r) => r.wellness_score);

  if (historyChart) {
    historyChart.data.labels = labels;
    historyChart.data.datasets[0].data = scores;
    historyChart.update();
    return;
  }

  historyChart = new Chart(chartCanvas.getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Wellness score",
        data: scores,
        borderColor: "#3f7a5d",
        backgroundColor: "rgba(63, 122, 93, 0.12)",
        tension: 0.3,
        fill: true,
        pointRadius: 3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 100, ticks: { stepSize: 20 } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/history`);
    if (!res.ok) throw new Error("Failed to load history.");
    const history = await res.json();
    renderChart(history);
  } catch (err) {
    chartCanvas.parentElement.hidden = true;
    chartEmpty.hidden = false;
    chartEmpty.textContent = "Couldn't load your history right now.";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showFormError("");

  let payload;
  try {
    payload = collectFormData();
  } catch (err) {
    showFormError(err.message);
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing...";

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || "The server could not process today's entry.");
    }

    const result = await res.json();
    renderResult(result);
    await loadHistory();
  } catch (err) {
    showFormError(err.message || "Something went wrong. Please try again.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Analyze My Day";
  }
});

loadHistory();
