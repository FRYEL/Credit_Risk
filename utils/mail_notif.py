import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os


def send_email(subject, body, to_email):
    """
    function to send email
    :param subject:
    :param body:
    :param to_email:
    :return:
    """
    # Email configurations

    # Load environment variables from .env file
    load_dotenv()

    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT'))  # Convert port to integer
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach body to the email
    msg.attach(MIMEText(body, 'plain'))

    # Create SMTP session
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()  # Enable TLS
    server.login(smtp_user, smtp_password)  # Login to email server

    # Send email
    server.sendmail(smtp_user, to_email, msg.as_string())

    # Quit server
    server.quit()

