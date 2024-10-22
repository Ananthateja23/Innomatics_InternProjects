# import required libraries
from flask import Flask, render_template, url_for, request
import re


# creation of flask app instance and passing __main__
app = Flask(__name__)

# home endpoint creation
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', title="Regex Matcher")


# results endpoint creation
@app.route("/results", methods=['POST'])
def results():
    test_string = request.form['testString']
    regex_pattern = request.form['regexPattern']

    try:
        # Compile the regex pattern
        pattern = re.compile(regex_pattern)
        # Find all matches in the test string
        matches = pattern.findall(test_string)
    except re.error:
        # Handle invalid regex patterns
        matches = ["Invalid regex pattern"]

    # Pass the list of matches to the template
    return render_template('home.html', matches=matches, title="Regex Matcher")

# New route for email validation
@app.route('/validate_email', methods=['GET', 'POST'])
def validate_email():
    title = "Email Validator"
    is_valid = None
    email = None
    if request.method == 'POST':
        email = request.form['email']
        # Regular expression for validating email
        email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if re.match(email_regex, email):
            is_valid = True
        else:
            is_valid = False
    return render_template('validate_email.html', title=title, is_valid=is_valid, email=email)



# To start the flask app
if __name__ == "__main__":
    app.run(debug=True)