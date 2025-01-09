let questions = [
    {
        question: "What is HTML?",
        answers: ["A programming language", "A markup language", "A style sheet", "None of these"],
        correct: 1
    },
    {
        question: "What does CSS stand for?",
        answers: ["Cascading Style Sheets", "Creative Style Sheets", "Computer Style Sheets", "None of these"],
        correct: 0
    }
];

let currentQuestion = 0;
let score = 0;

function loadQuestion() {
    document.getElementById("question").textContent = questions[currentQuestion].question;
    let buttons = document.querySelectorAll("#quizContainer button");
    buttons.forEach((button, index) => {
        button.textContent = questions[currentQuestion].answers[index];
    });
}

function submitAnswer(answerIndex) {
    if (answerIndex === questions[currentQuestion].correct) {
        score++;
    }
    currentQuestion++;
    if (currentQuestion < questions.length) {
        loadQuestion();
    } else {
        document.getElementById("quizContainer").style.display = "none";
        document.getElementById("score").textContent = `Your score: ${score}`;
    }
}

// Initialize quiz
window.onload = loadQuestion;
