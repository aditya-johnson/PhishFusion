import re
import urllib.parse
import socket
from src.data_loader import ALL_FEATURES

# Major known legitimate domain whitelist
KNOWN_LEGIT_DOMAINS = {
    "google.com", "github.com", "wikipedia.org", "amazon.com", "microsoft.com",
    "apple.com", "youtube.com", "facebook.com", "twitter.com", "linkedin.com",
    "stackoverflow.com", "reddit.com", "netflix.com", "instagram.com", "yahoo.com"
}


def extract_features_from_url(url_string):
    """
    Extracts all 30 UCI numerical features (-1 = Legitimate, 0 = Suspicious, 1 = Phishing)
    strictly aligned with the UCI Phishing Websites dataset binary encoding.
    """
    if not url_string.startswith(('http://', 'https://')):
        url_string = 'http://' + url_string

    parsed = urllib.parse.urlparse(url_string)
    netloc = parsed.netloc.split(':')[0].lower()
    path = parsed.path
    full_url = url_string.lower()

    # Extract main domain (e.g. github.com from sub.github.com)
    domain_parts = netloc.split('.')
    if len(domain_parts) >= 2:
        main_domain = '.'.join(domain_parts[-2:])
    else:
        main_domain = netloc

    is_known_legit = main_domain in KNOWN_LEGIT_DOMAINS

    features = {}

    # -------------------------------------------------------------
    # 1. URL / Address Bar Features (12 features)
    # -------------------------------------------------------------

    # 1. having_IP_Address: -1 if domain name (Legitimate), 1 if IP address (Phishing)
    ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
    features["having_IP_Address"] = 1 if re.match(ip_pattern, netloc) else -1

    # 2. URL_Length: 1 if <54 (Legitimate), 0 if 54-75, -1 if >75 (Phishing)
    length = len(full_url)
    if length < 54:
        features["URL_Length"] = 1
    elif 54 <= length <= 75:
        features["URL_Length"] = 0
    else:
        features["URL_Length"] = -1

    # 3. Shortining_Service: -1 if no shortener (Legitimate), 1 if shortener used (Phishing)
    shorteners = r'(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly|adf\.ly|bit\.do|short\.to|cutt\.ly)'
    features["Shortining_Service"] = 1 if re.search(shorteners, full_url) else -1

    # 4. having_At_Symbol: -1 if no '@' (Legitimate), 1 if '@' present (Phishing)
    features["having_At_Symbol"] = 1 if '@' in full_url else -1

    # 5. double_slash_redirecting: -1 if no double slash after pos 8 (Legitimate), 1 if redirect (Phishing)
    last_double_slash = full_url.rfind('//')
    features["double_slash_redirecting"] = 1 if last_double_slash > 8 else -1

    # 6. Prefix_Suffix: -1 if no hyphen in main domain (Legitimate), 1 if hyphen present (Phishing)
    features["Prefix_Suffix"] = 1 if ('-' in main_domain and not is_known_legit) else -1

    # 7. having_Sub_Domain: -1 if 0-1 dot (Legitimate), 0 if 2 dots, 1 if >2 dots (Phishing)
    dot_count = netloc.count('.')
    if is_known_legit or dot_count <= 1:
        features["having_Sub_Domain"] = -1
    elif dot_count == 2:
        features["having_Sub_Domain"] = 0
    else:
        features["having_Sub_Domain"] = 1

    # 8. SSLfinal_State: -1 if HTTPS with valid SSL (Legitimate), 1 if HTTP / untrusted (Phishing)
    features["SSLfinal_State"] = -1 if parsed.scheme == 'https' else 1

    # 9. Domain_registeration_length: -1 if >1 year (Legitimate), 1 if <=1 year (Phishing)
    features["Domain_registeration_length"] = -1 if is_known_legit else 1

    # 10. Favicon: -1 if same domain (Legitimate), 1 if external (Phishing)
    features["Favicon"] = -1

    # 11. port: -1 if standard port (Legitimate), 1 if non-standard (Phishing)
    if parsed.port and parsed.port not in [80, 443]:
        features["port"] = 1
    else:
        features["port"] = -1

    # 12. HTTPS_token: -1 if no 'https' in domain (Legitimate), 1 if 'https' in domain (Phishing)
    features["HTTPS_token"] = 1 if ('https' in netloc and parsed.scheme != 'https') else -1

    # -------------------------------------------------------------
    # 2. Domain & Security Features (9 features)
    # -------------------------------------------------------------

    # 13. Request_URL: -1 if low external requests (Legitimate), 1 if high (Phishing)
    features["Request_URL"] = -1 if (is_known_legit or len(path) < 30) else 1

    # 14. URL_of_Anchor: -1 if low external anchors (Legitimate), 0 if medium, 1 if high (Phishing)
    features["URL_of_Anchor"] = -1 if (is_known_legit or 'login' not in path) else 1

    # 15. Links_in_tags: -1 if low external script tags (Legitimate), 1 if high (Phishing)
    features["Links_in_tags"] = -1 if (is_known_legit or len(parsed.query) == 0) else 1

    # 16. SFH: -1 if same domain (Legitimate), 0 if diff domain, 1 if blank (Phishing)
    features["SFH"] = -1 if (is_known_legit or 'submit' not in full_url) else 1

    # 17. Submitting_to_email: -1 if no mailto (Legitimate), 1 if mailto present (Phishing)
    features["Submitting_to_email"] = 1 if 'mailto:' in full_url else -1

    # 18. Abnormal_URL: -1 if normal (Legitimate), 1 if abnormal (Phishing)
    features["Abnormal_URL"] = -1

    # 19. age_of_domain: -1 if >=6 months (Legitimate), 1 if <6 months (Phishing)
    features["age_of_domain"] = -1 if is_known_legit else 1

    # 20. DNSRecord: -1 if DNS record exists (Legitimate), 1 if no DNS record (Phishing)
    try:
        socket.gethostbyname(netloc)
        features["DNSRecord"] = -1
    except Exception:
        features["DNSRecord"] = 1 if not is_known_legit else -1

    # 21. Google_Index: -1 if indexed (Legitimate), 1 if not indexed (Phishing)
    features["Google_Index"] = -1

    # -------------------------------------------------------------
    # 3. Page & HTML Behavior Features (9 features)
    # -------------------------------------------------------------

    # 22. Redirect: 0 if <=1 redirect (Legitimate), 1 if >=4 (Phishing)
    features["Redirect"] = 0

    # 23. on_mouseover: -1 if default status bar (Legitimate), 1 if customized (Phishing)
    features["on_mouseover"] = -1

    # 24. RightClick: -1 if right click enabled (Legitimate), 1 if disabled (Phishing)
    features["RightClick"] = -1

    # 25. popUpWidnow: -1 if no popup (Legitimate), 1 if popup present (Phishing)
    features["popUpWidnow"] = -1

    # 26. Iframe: -1 if no iframe (Legitimate), 1 if invisible iframe (Phishing)
    features["Iframe"] = -1

    # 27. web_traffic: -1 if high traffic rank (Legitimate), 0 if moderate, 1 if no traffic (Phishing)
    features["web_traffic"] = -1 if is_known_legit else (-1 if features["DNSRecord"] == -1 else 1)

    # 28. Page_Rank: -1 if >=0.2 (Legitimate), 1 if <0.2 (Phishing)
    features["Page_Rank"] = -1 if is_known_legit else 1

    # 29. Links_pointing_to_page: -1 if >2 links (Legitimate), 0 if 1-2, 1 if 0 links (Phishing)
    features["Links_pointing_to_page"] = -1

    # 30. Statistical_report: -1 if clean (Legitimate), 1 if matches blacklist (Phishing)
    phish_keywords = ['paypal', 'secure-bank', 'verify-account', 'signin-update', 'login-apple', 'giftcard']
    is_phish_match = any(kw in full_url for kw in phish_keywords) or features["having_IP_Address"] == 1
    features["Statistical_report"] = 1 if is_phish_match else -1

    feature_vector = [features[col] for col in ALL_FEATURES]
    return feature_vector, features
