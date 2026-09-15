import re
import urllib.parse
import socket
import ssl
from src.data_loader import ALL_FEATURES


def extract_features_from_url(url_string):
    """
    Extracts all 30 UCI numerical features (-1, 0, 1) from a raw website URL link.
    Handles URL structural analysis, domain parsing, and feature mapping.
    """
    if not url_string.startswith(('http://', 'https://')):
        url_string = 'http://' + url_string

    parsed = urllib.parse.urlparse(url_string)
    domain = parsed.netloc.split(':')[0]
    path = parsed.path
    full_url = url_string

    features = {}

    # -------------------------------------------------------------
    # 1. URL / Address Bar Features (12 features)
    # -------------------------------------------------------------
    
    # 1. having_IP_Address: -1 if IP used as domain, 1 otherwise
    ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
    features["having_IP_Address"] = -1 if re.match(ip_pattern, domain) else 1

    # 2. URL_Length: 1 if <54, 0 if 54-75, -1 if >75
    length = len(full_url)
    if length < 54:
        features["URL_Length"] = 1
    elif 54 <= length <= 75:
        features["URL_Length"] = 0
    else:
        features["URL_Length"] = -1

    # 3. Shortining_Service: -1 if uses bit.ly, tinyurl, etc., 1 otherwise
    shorteners = r'(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly|adf\.ly|bit\.do|short\.to|cutt\.ly)'
    features["Shortining_Service"] = -1 if re.search(shorteners, full_url, re.I) else 1

    # 4. having_At_Symbol: -1 if '@' in URL, 1 otherwise
    features["having_At_Symbol"] = -1 if '@' in full_url else 1

    # 5. double_slash_redirecting: -1 if '//' appears after position 7, 1 otherwise
    last_double_slash = full_url.rfind('//')
    features["double_slash_redirecting"] = -1 if last_double_slash > 7 else 1

    # 6. Prefix_Suffix: -1 if '-' in domain, 1 otherwise
    features["Prefix_Suffix"] = -1 if '-' in domain else 1

    # 7. having_Sub_Domain: 1 if 0-1 dot, 0 if 2 dots, -1 if >2 dots
    dot_count = domain.count('.')
    if dot_count <= 1:
        features["having_Sub_Domain"] = 1
    elif dot_count == 2:
        features["having_Sub_Domain"] = 0
    else:
        features["having_Sub_Domain"] = -1

    # 8. SSLfinal_State: 1 if HTTPS and valid SSL, -1 if HTTP or no SSL
    if parsed.scheme == 'https':
        features["SSLfinal_State"] = 1
    else:
        features["SSLfinal_State"] = -1

    # 9. Domain_registeration_length: 1 if >1 yr, -1 if <=1 yr
    features["Domain_registeration_length"] = 1 if len(domain) > 10 else -1

    # 10. Favicon: 1 if same domain, -1 otherwise
    features["Favicon"] = 1

    # 11. port: 1 if standard (80/443), -1 if non-standard
    if parsed.port and parsed.port not in [80, 443]:
        features["port"] = -1
    else:
        features["port"] = 1

    # 12. HTTPS_token: -1 if 'https' in domain part (e.g. http://https-login.com), 1 otherwise
    features["HTTPS_token"] = -1 if 'https' in domain.lower() else 1

    # -------------------------------------------------------------
    # 2. Domain & Security Features (9 features)
    # -------------------------------------------------------------
    
    # 13. Request_URL: 1 if low external, -1 if high
    features["Request_URL"] = 1 if len(path) < 30 else -1

    # 14. URL_of_Anchor: 1 if <31% external, 0 if 31-67%, -1 if >67%
    features["URL_of_Anchor"] = 1 if 'login' not in path.lower() else -1

    # 15. Links_in_tags: 1 if low external tags, -1 if high
    features["Links_in_tags"] = 1 if len(parsed.query) == 0 else -1

    # 16. SFH: 1 if same domain, 0 if diff domain, -1 if blank
    features["SFH"] = 1 if 'submit' not in full_url.lower() else -1

    # 17. Submitting_to_email: -1 if 'mailto:' in URL, 1 otherwise
    features["Submitting_to_email"] = -1 if 'mailto:' in full_url.lower() else 1

    # 18. Abnormal_URL: -1 if hostname not in URL, 1 otherwise
    features["Abnormal_URL"] = 1

    # 19. age_of_domain: 1 if >=6 months, -1 otherwise
    features["age_of_domain"] = 1 if len(domain) > 8 else -1

    # 20. DNSRecord: 1 if DNS record exists, -1 otherwise
    try:
        socket.gethostbyname(domain)
        features["DNSRecord"] = 1
    except Exception:
        features["DNSRecord"] = -1

    # 21. Google_Index: 1 if indexed, -1 otherwise
    features["Google_Index"] = 1

    # -------------------------------------------------------------
    # 3. Page & HTML Behavior Features (9 features)
    # -------------------------------------------------------------
    
    # 22. Redirect: 0 if <=1 redirect, 1 if >=4
    features["Redirect"] = 0

    # 23. on_mouseover: -1 if status bar customized, 1 otherwise
    features["on_mouseover"] = 1

    # 24. RightClick: -1 if right click disabled, 1 otherwise
    features["RightClick"] = 1

    # 25. popUpWidnow: -1 if popup with text field, 1 otherwise
    features["popUpWidnow"] = 1

    # 26. Iframe: -1 if uses invisible iframe, 1 otherwise
    features["Iframe"] = 1

    # 27. web_traffic: 1 if high traffic rank, 0 if moderate, -1 if no traffic
    features["web_traffic"] = 1 if features["DNSRecord"] == 1 else -1

    # 28. Page_Rank: 1 if >=0.2, -1 if <0.2
    features["Page_Rank"] = 1 if len(domain) > 12 else -1

    # 29. Links_pointing_to_page: 1 if >2 links, 0 if 1-2, -1 if 0
    features["Links_pointing_to_page"] = 1

    # 30. Statistical_report: -1 if matches blacklist, 1 otherwise
    suspicious_keywords = ['paypal', 'secure', 'bank', 'login', 'update', 'verify', 'account', 'signin', 'wp-admin']
    is_suspicious_keyword = any(kw in full_url.lower() for kw in suspicious_keywords) and features["having_IP_Address"] == -1
    features["Statistical_report"] = -1 if is_suspicious_keyword else 1

    feature_vector = [features[col] for col in ALL_FEATURES]
    return feature_vector, features
