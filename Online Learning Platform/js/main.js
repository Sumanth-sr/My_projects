// JavaScript for handling register, login, and redirecting users

// Register function (saving user data in localStorage for demo)
function registerUser(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    // Save user to localStorage
    localStorage.setItem('username', username);
    localStorage.setItem('password', password);

    alert('Registration successful! You can now log in.');
    window.location.href = 'login.html'; // Redirect to login
}

// Login function
function loginUser(event) {
    event.preventDefault();
    const loginUsername = document.getElementById('loginUsername').value;
    const loginPassword = document.getElementById('loginPassword').value;

    // Check if username and password match
    const savedUsername = localStorage.getItem('username');
    const savedPassword = localStorage.getItem('password');

    if (loginUsername === savedUsername && loginPassword === savedPassword) {
        alert('Login successful!');
        localStorage.setItem('isLoggedIn', 'true');  // Set session
        window.location.href = 'dashboard.html';    // Redirect to dashboard
    } else {
        alert('Incorrect username or password. Please try again.');
    }
}

// Logout function (clear session)
function logoutUser() {
    localStorage.removeItem('isLoggedIn'); // Clear session
    alert('You have been logged out.');
    window.location.href = 'index.html'; // Redirect to Home
}

// Check if the user is logged in (use this for page protection)
function checkLoginStatus() {
    const isLoggedIn = localStorage.getItem('isLoggedIn');
    if (!isLoggedIn) {
        window.location.href = 'login.html'; // Redirect to login if not logged in
    }
}
// Check login status (already provided)
function checkLoginStatus() {
    const isLoggedIn = localStorage.getItem('isLoggedIn');
    if (!isLoggedIn) {
        window.location.href = 'login.html'; // Redirect to login if not logged in
    }
}

// Submit quiz and calculate score
function submitQuiz(event) {
    event.preventDefault();

    // Get form data
    const form = event.target;
    const formData = new FormData(form);

    // Define correct answers
    const correctAnswers = {
        q1: 'c', // Paris
        q2: 'b'  // 4
    };

    let score = 0;
    let correctCount = 0;
    let wrongCount = 0;

    // Process each answer and show feedback
    for (const [question, answer] of formData.entries()) {
        const feedbackElement = document.getElementById(`feedback-${question}`);
        if (correctAnswers[question] === answer) {
            score++;
            correctCount++;
            feedbackElement.innerHTML = `Correct! The answer is ${correctAnswers[question].toUpperCase()}.`;
            feedbackElement.style.color = "green";
        } else {
            wrongCount++;
            feedbackElement.innerHTML = `Wrong! The correct answer is ${correctAnswers[question].toUpperCase()}.`;
            feedbackElement.style.color = "red";
        }
    }

    // Display result
    const resultElement = document.getElementById('result');
    resultElement.textContent = `You scored ${score} out of ${Object.keys(correctAnswers).length}.`;

    // Update user statistics in localStorage
    updateUserProfile(correctCount, wrongCount);
}

// Update user's profile with quiz results
function updateUserProfile(correct, wrong) {
    const totalQuizzes = parseInt(localStorage.getItem('totalQuizzes')) || 0;
    const totalCorrect = parseInt(localStorage.getItem('totalCorrect')) || 0;
    const totalWrong = parseInt(localStorage.getItem('totalWrong')) || 0;

    localStorage.setItem('totalQuizzes', totalQuizzes + 1);
    localStorage.setItem('totalCorrect', totalCorrect + correct);
    localStorage.setItem('totalWrong', totalWrong + wrong);
}

// Display user profile information on profile.html
function displayUserProfile() {
    const username = localStorage.getItem('username');
    const totalQuizzes = localStorage.getItem('totalQuizzes') || 0;
    const totalCorrect = localStorage.getItem('totalCorrect') || 0;
    const totalWrong = localStorage.getItem('totalWrong') || 0;

    document.getElementById('usernameDisplay').textContent = username;
    document.getElementById('totalQuizzes').textContent = totalQuizzes;
    document.getElementById('totalCorrect').textContent = totalCorrect;
    document.getElementById('totalWrong').textContent = totalWrong;
}
