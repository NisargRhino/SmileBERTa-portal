---
layout: none
permalink: /home
title: SmileBERTa Home
---

<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to SmileBERTa</title>
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121212; /* Dark background */
            color: #ffffff; /* White text */
            scroll-behavior: smooth; /* Enables smooth scrolling */
        }

        /* Navigation Bar */
        .navbar {
            background-color: #1e1e1e;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 30px;
            position: sticky;
            top: 0;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
        }

        .navbar a {
            color: white;
            text-decoration: none;
            margin: 0 15px;
            font-size: 18px;
            font-weight: 500;
            transition: color 0.3s;
        }

        .navbar a:hover {
            color: #1e90ff;
        }

        .logout-btn {
            background-color: #ff4d4d;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }

        .logout-btn:hover {
            background-color: #e43e3e;
        }

        /* Section Styling */
        .section {
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 20px;
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
            text-shadow: 0 0 10px rgba(0, 0, 0, 0.7); /* Subtle shadow for better readability */
        }

        .section h1 {
            font-size: 3.5rem;
            margin-bottom: 20px;
            font-weight: bold;
        }

        .section p {
            font-size: 1.3rem;
            margin-bottom: 30px;
            line-height: 1.6;
        }

        .blue-theme a {
            color: #1e90ff;
            text-decoration: none;
            font-weight: bold;
            font-size: 1.2rem;
        }

        .blue-theme a:hover {
            text-decoration: underline;
            color: #00bfff;
        }

        /* Backgrounds for each section */
        .section-1 {
            background-image: url('./picture1.jpg');
            background-color: #1e1e1e; /* Fallback color */
        }

        .section-2 {
            background-image: url('./picture2.jpg');
            background-color: #222222; /* Fallback color */
        }

        .section-3 {
            background-image: url('./picture3.jpg');
            background-color: #1e1e1e; /* Fallback color */
        }

        .section-4 {
            background-image: url('./picture4.png');
            background-color: #222222; /* Fallback color */
        }
    </style>
</head>
<body>
    <!-- Navigation Bar -->
    <nav class="navbar">
        <div class="navbar-left">
            <a href="#">SmileBERTa</a>
        </div>
        <div class="navbar-right">
            <a href="mol_editing.html">Molecular Editing</a>
            <a href="fragmentation.html">Fragmentation</a>
            <a href="drug_class.html">Drug Classification</a>
            <a href="#section-4">About</a>
        </div>
    </nav>

    <!-- Section 1 -->
    <div class="section section-1" id="section-1">
        <div>
            <h1>Welcome to SmileBERTa</h1>
            <p>Welcome to our powerful platform for accelerating drug discovery. Explore our tools to bring your ideas to life.</p>
            <a class="blue-theme" href="mol_editing.html">Start with Molecular Editing</a>
        </div>
    </div>

    <!-- Section 2 -->
    <div class="section section-2" id="section-2">
        <div>
            <h1>Fragmentation</h1>
            <p>Effortlessly break down complex molecules with our state-of-the-art fragmentation tools. Analyze your data faster and smarter.</p>
            <a class="blue-theme" href="fragmentation.html">Learn More about Fragmentation</a>
        </div>
    </div>

    <!-- Section 3 -->
    <div class="section section-3" id="section-3">
        <div>
            <h1>Drug Classification</h1>
            <p>Classify and analyze the properties of drugs using our advanced algorithms. Make data-driven decisions with ease.</p>
            <a class="blue-theme" href="drug_class.html">Explore Drug Classification</a>
        </div>
    </div>

    <!-- Section 4 -->
    <div class="section section-4" id="section-4">
        <div>
            <h1>About Us</h1>
            <p>SmileBERTa is revolutionizing fragment-based drug discovery with cutting-edge AI tools. Learn more about our mission and vision.
                
            </p>
            <p>Contact Info: Nisarg Shah 11th Grade at Del Norte High School </p>
            <p>Email: nisargs2018@gmail.com</p>
            
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>