
// Register new user
function registerUser(event) {
    event.preventDefault();
    let username = document.getElementById('username').value;
    let password = document.getElementById('password').value;

    if (!localStorage.getItem(username)) {
        localStorage.setItem(username, password);
        alert("User registered successfully!");
    } else {
        alert("Username already exists.");
    }
}

// Login existing user
function loginUser(event) {
    event.preventDefault();
    let username = document.getElementById('loginUsername').value;
    let password = document.getElementById('loginPassword').value;

    if (localStorage.getItem(username) === password) {
        alert("Login successful!");
        window.location.href = "dashboard.html";  // Redirect to dashboard
    } else {
        alert("Invalid credentials.");
    }
}
