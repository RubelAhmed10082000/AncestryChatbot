const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const resetButton = document.getElementById("reset-button");

const questions = [
  {
    key: "first_name",
    text: "What is the person's first name?",
    required: true,
  },
  {
    key: "last_name",
    text: "What is their last name at birth or family surname?",
    required: true,
  },
  {
    key: "birth_year",
    text: "What birth year should I use? You can leave this blank if unknown.",
    required: false,
  },
  {
    key: "birth_location",
    text: "Where were they born? You can leave this blank if unknown.",
    required: false,
  },
  {
    key: "gender",
    text: "What gender should I use? Enter Male, Female, or leave blank.",
    required: false,
  },
];

let stepIndex = 0;
let profile = {};

function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  chatWindow.appendChild(message);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return message;
}

function addHtmlMessage(role, element) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  wrapper.appendChild(element);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function askCurrentQuestion() {
  const question = questions[stepIndex];
  addMessage("assistant", question.text);
}

function normaliseAnswer(question, rawAnswer) {
  const answer = rawAnswer.trim();

  if (!answer) {
    return null;
  }

  if (question.key === "birth_year") {
    const parsed = Number.parseInt(answer, 10);
    return Number.isNaN(parsed) ? null : parsed;
  }

  return answer;
}

async function searchCandidates() {
  addMessage("assistant", "Searching for candidate profiles...");

  const requestBody = {
    first_name: profile.first_name,
    last_name: profile.last_name,
    birth_year: profile.birth_year,
    birth_location: profile.birth_location,
    gender: profile.gender,
    top_k: 5,
    min_score: 0,
  };

  try {
    const response = await fetch("/api/candidates/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Candidate search failed.");
    }

    const data = await response.json();
    renderCandidates(data);
  } catch (error) {
    addMessage("assistant", `Something went wrong: ${error.message}`);
  }
}

function renderCandidates(data) {
  if (!data.candidates || data.candidates.length === 0) {
    addMessage(
      "assistant",
      "No candidates were found. Try again with fewer constraints or a broader location."
    );
    return;
  }

  const container = document.createElement("div");

  const heading = document.createElement("p");
  heading.textContent = `I found ${data.count} candidate(s). Here are the best matches:`;
  container.appendChild(heading);

  data.candidates.forEach((candidate) => {
    const card = document.createElement("div");
    card.className = "candidate-card";

    const title = document.createElement("h3");
    title.textContent = `#${candidate.rank} — ${candidate.full_name || "Unknown name"}`;
    card.appendChild(title);

    const meta = document.createElement("p");
    meta.className = "meta";
    meta.textContent = [
      candidate.wikitree_id ? `WikiTree ID: ${candidate.wikitree_id}` : null,
      candidate.birth_year ? `Born: ${candidate.birth_year}` : null,
      candidate.birth_location ? `Location: ${candidate.birth_location}` : null,
    ]
      .filter(Boolean)
      .join(" | ");
    card.appendChild(meta);

    const confidence = document.createElement("p");
    confidence.className = "confidence";
    confidence.textContent = `Confidence: ${
      candidate.confidence_score ?? "N/A"
    } | Rank score: ${candidate.rank_score ?? "N/A"}`;
    card.appendChild(confidence);

    if (candidate.confidence_explanation) {
      const explanation = document.createElement("p");
      explanation.textContent = candidate.confidence_explanation;
      card.appendChild(explanation);
    }

    if (candidate.confidence_interpretation) {
      const interpretation = document.createElement("p");
      interpretation.className = "meta";
      interpretation.textContent = candidate.confidence_interpretation;
      card.appendChild(interpretation);
    }

    if (candidate.wikitree_id) {
      const treeButton = document.createElement("button");
      treeButton.className = "card-button";
      treeButton.textContent = "View family tree";
      treeButton.addEventListener("click", () => {
        loadTree(candidate.wikitree_id);
      });
      card.appendChild(treeButton);
    }

    container.appendChild(card);
  });

  addHtmlMessage("assistant", container);
}

async function loadTree(wikitreeId) {
  addMessage("assistant", `Generating family tree for ${wikitreeId}...`);

  try {
    const response = await fetch(
      `/api/tree/by-wikitree/${encodeURIComponent(wikitreeId)}?generations=3`
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Tree generation failed.");
    }

    const tree = await response.json();
    renderTree(tree);
  } catch (error) {
    addMessage("assistant", `Something went wrong: ${error.message}`);
  }
}

function renderTree(tree) {
  const panel = document.createElement("div");
  panel.className = "tree-panel";

  const title = document.createElement("h3");
  title.textContent = "Preliminary family tree";
  panel.appendChild(title);

  const summary = document.createElement("p");
  summary.className = "meta";
  summary.textContent = `Root ID: ${tree.root_person_id} | Generations: ${tree.generations}`;
  panel.appendChild(summary);

  const nodesByGeneration = {};

  tree.nodes.forEach((node) => {
    const generation = node.generation ?? 0;
    if (!nodesByGeneration[generation]) {
      nodesByGeneration[generation] = [];
    }
    nodesByGeneration[generation].push(node);
  });

  Object.keys(nodesByGeneration)
    .sort((a, b) => Number(a) - Number(b))
    .forEach((generation) => {
      const section = document.createElement("div");
      section.className = "tree-generation";

      const heading = document.createElement("h4");
      heading.textContent =
        Number(generation) === 0
          ? "Generation 0 — root person"
          : `Generation ${generation} — ancestor level ${generation}`;
      section.appendChild(heading);

      nodesByGeneration[generation].forEach((node) => {
        const person = document.createElement("p");
        person.className = "tree-person";
        person.textContent = `${node.full_name || "Unknown"} ${
          node.wikitree_id ? `(${node.wikitree_id})` : ""
        } ${node.birth_year ? `— born ${node.birth_year}` : ""}`;
        section.appendChild(person);
      });

      panel.appendChild(section);
    });

  addHtmlMessage("assistant", panel);
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const answer = chatInput.value;
  if (!answer.trim() && questions[stepIndex]?.required) {
    addMessage("assistant", "I need this field before we can continue.");
    return;
  }

  addMessage("user", answer || "[blank]");
  chatInput.value = "";

  const question = questions[stepIndex];
  profile[question.key] = normaliseAnswer(question, answer);

  stepIndex += 1;

  if (stepIndex < questions.length) {
    askCurrentQuestion();
  } else {
    await searchCandidates();
  }
});

resetButton.addEventListener("click", () => {
  stepIndex = 0;
  profile = {};
  chatWindow.innerHTML = "";
  addMessage(
    "assistant",
    "Hi — I’ll ask a few questions and then search for probable ancestor candidates."
  );
  askCurrentQuestion();
});

resetButton.click();