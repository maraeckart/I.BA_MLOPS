import argparse
from datetime import datetime
from pathlib import Path
import re

import pandas as pd

REGION_KEYWORDS = {
    "World": ["world", "global", "international", "worldwide"],
    "Europe": ["europe", "european"],
    "European Union": ["european union", "eu", "brussels"],
    "Africa": ["africa", "african"],
    "Asia": ["asia", "asian"],
    "Middle East": ["middle east", "middle eastern"],
    "North America": ["north america", "north american"],
    "South America": ["south america", "south american"],
    "Latin America": ["latin america", "latin american"],
    "Oceania": ["oceania"],
    "Arctic": ["arctic"],
    "Antarctica": ["antarctica", "antarctic"],

    "Albania": ["albania", "albanian"],
    "Andorra": ["andorra", "andorran"],
    "Austria": ["austria", "austrian", "vienna"],
    "Belarus": ["belarus", "belarusian", "minsk"],
    "Belgium": ["belgium", "belgian"],
    "Bosnia and Herzegovina": ["bosnia and herzegovina", "bosnia", "bosnian", "herzegovina"],
    "Bulgaria": ["bulgaria", "bulgarian", "sofia"],
    "Croatia": ["croatia", "croatian", "zagreb"],
    "Cyprus": ["cyprus", "cypriot"],
    "Czech Republic": ["czech republic", "czechia", "czech", "prague"],
    "Denmark": ["denmark", "danish", "copenhagen"],
    "Estonia": ["estonia", "estonian", "tallinn"],
    "Finland": ["finland", "finnish", "helsinki"],
    "France": ["france", "french", "paris"],
    "Germany": ["germany", "german", "berlin"],
    "Greece": ["greece", "greek", "athens"],
    "Hungary": ["hungary", "hungarian", "budapest"],
    "Iceland": ["iceland", "icelandic", "reykjavik"],
    "Ireland": ["ireland", "irish", "dublin"],
    "Italy": ["italy", "italian", "rome"],
    "Kosovo": ["kosovo"],
    "Latvia": ["latvia", "latvian", "riga"],
    "Liechtenstein": ["liechtenstein"],
    "Lithuania": ["lithuania", "lithuanian", "vilnius"],
    "Luxembourg": ["luxembourg"],
    "Malta": ["malta", "maltese"],
    "Moldova": ["moldova", "moldovan"],
    "Monaco": ["monaco"],
    "Montenegro": ["montenegro"],
    "Netherlands": ["netherlands", "dutch", "amsterdam"],
    "North Macedonia": ["north macedonia", "macedonia", "macedonian"],
    "Norway": ["norway", "norwegian", "oslo"],
    "Poland": ["poland", "polish", "warsaw"],
    "Portugal": ["portugal", "portuguese", "lisbon"],
    "Romania": ["romania", "romanian", "bucharest"],
    "Russia": ["russia", "russian", "moscow"],
    "San Marino": ["san marino"],
    "Serbia": ["serbia", "serbian", "belgrade"],
    "Slovakia": ["slovakia", "slovak", "bratislava"],
    "Slovenia": ["slovenia", "slovenian", "ljubljana"],
    "Spain": ["spain", "spanish", "madrid"],
    "Sweden": ["sweden", "swedish", "stockholm"],
    "Switzerland": ["switzerland", "swiss", "zurich", "geneva", "bern"],
    "Turkey": ["turkey", "turkish", "ankara", "istanbul"],
    "Ukraine": ["ukraine", "ukrainian", "kyiv", "kiev"],
    "United Kingdom": [
        "united kingdom",
        "britain",
        "british",
        "england",
        "scotland",
        "wales",
        "northern ireland",
        "london",
    ],
    "Vatican City": ["vatican", "vatican city"],

    "United States": [
        "united states",
        "us",
        "u.s.",
        "usa",
        "america",
        "american",
        "washington",
        "new york",
        "california",
        "texas",
        "florida",
    ],
    "Canada": ["canada", "canadian", "ottawa", "toronto", "vancouver", "montreal"],
    "Mexico": ["mexico", "mexican", "mexico city"],

    "Belize": ["belize"],
    "Costa Rica": ["costa rica", "costa rican"],
    "El Salvador": ["el salvador", "salvadoran"],
    "Guatemala": ["guatemala", "guatemalan"],
    "Honduras": ["honduras", "honduran"],
    "Nicaragua": ["nicaragua", "nicaraguan"],
    "Panama": ["panama", "panamanian"],
    "Cuba": ["cuba", "cuban", "havana"],
    "Haiti": ["haiti", "haitian"],
    "Dominican Republic": ["dominican republic", "dominican"],
    "Jamaica": ["jamaica", "jamaican"],
    "Bahamas": ["bahamas"],
    "Barbados": ["barbados"],
    "Trinidad and Tobago": ["trinidad and tobago", "trinidad", "tobago"],

    "Argentina": ["argentina", "argentinian", "argentine", "buenos aires"],
    "Bolivia": ["bolivia", "bolivian"],
    "Brazil": ["brazil", "brazilian", "brasilia", "rio de janeiro", "sao paulo"],
    "Chile": ["chile", "chilean", "santiago"],
    "Colombia": ["colombia", "colombian", "bogota"],
    "Ecuador": ["ecuador", "ecuadorian", "quito"],
    "Guyana": ["guyana"],
    "Paraguay": ["paraguay", "paraguayan"],
    "Peru": ["peru", "peruvian", "lima"],
    "Suriname": ["suriname"],
    "Uruguay": ["uruguay", "uruguayan"],
    "Venezuela": ["venezuela", "venezuelan", "caracas"],

    "Iran": ["iran", "iranian", "tehran"],
    "Iraq": ["iraq", "iraqi", "baghdad"],
    "Israel": ["israel", "israeli", "jerusalem", "tel aviv"],
    "Palestine": ["palestine", "palestinian", "gaza", "west bank"],
    "Lebanon": ["lebanon", "lebanese", "beirut"],
    "Syria": ["syria", "syrian", "damascus"],
    "Jordan": ["jordan", "jordanian", "amman"],
    "Saudi Arabia": ["saudi arabia", "saudi", "riyadh"],
    "United Arab Emirates": ["united arab emirates", "uae", "dubai", "abu dhabi"],
    "Qatar": ["qatar", "qatari", "doha"],
    "Kuwait": ["kuwait", "kuwaiti"],
    "Bahrain": ["bahrain", "bahraini"],
    "Oman": ["oman", "omani"],
    "Yemen": ["yemen", "yemeni"],

    "Afghanistan": ["afghanistan", "afghan", "kabul"],
    "Armenia": ["armenia", "armenian", "yerevan"],
    "Azerbaijan": ["azerbaijan", "azerbaijani", "baku"],
    "Bangladesh": ["bangladesh", "bangladeshi", "dhaka"],
    "Bhutan": ["bhutan", "bhutanese"],
    "Cambodia": ["cambodia", "cambodian", "phnom penh"],
    "China": ["china", "chinese", "beijing", "shanghai", "hong kong"],
    "Georgia": ["georgia", "georgian", "tbilisi"],
    "India": ["india", "indian", "delhi", "new delhi", "mumbai"],
    "Indonesia": ["indonesia", "indonesian", "jakarta", "bali"],
    "Japan": ["japan", "japanese", "tokyo"],
    "Kazakhstan": ["kazakhstan", "kazakh", "astana"],
    "Kyrgyzstan": ["kyrgyzstan", "kyrgyz"],
    "Laos": ["laos", "laotian"],
    "Malaysia": ["malaysia", "malaysian", "kuala lumpur"],
    "Maldives": ["maldives"],
    "Mongolia": ["mongolia", "mongolian"],
    "Myanmar": ["myanmar", "burma", "burmese"],
    "Nepal": ["nepal", "nepalese", "kathmandu"],
    "North Korea": ["north korea", "north korean", "pyongyang"],
    "Pakistan": ["pakistan", "pakistani", "islamabad", "karachi"],
    "Philippines": ["philippines", "philippine", "filipino", "manila"],
    "Singapore": ["singapore", "singaporean"],
    "South Korea": ["south korea", "south korean", "seoul"],
    "Sri Lanka": ["sri lanka", "sri lankan", "colombo"],
    "Taiwan": ["taiwan", "taiwanese", "taipei"],
    "Tajikistan": ["tajikistan", "tajik"],
    "Thailand": ["thailand", "thai", "bangkok"],
    "Turkmenistan": ["turkmenistan", "turkmen"],
    "Uzbekistan": ["uzbekistan", "uzbek"],
    "Vietnam": ["vietnam", "vietnamese", "hanoi", "ho chi minh"],

    "Algeria": ["algeria", "algerian", "algiers"],
    "Angola": ["angola", "angolan"],
    "Botswana": ["botswana"],
    "Cameroon": ["cameroon", "cameroonian"],
    "Democratic Republic of the Congo": [
        "democratic republic of the congo",
        "dr congo",
        "drc",
        "congolese",
        "kinshasa",
    ],
    "Republic of the Congo": ["republic of the congo", "congo-brazzaville", "brazzaville"],
    "Egypt": ["egypt", "egyptian", "cairo"],
    "Ethiopia": ["ethiopia", "ethiopian", "addis ababa"],
    "Ghana": ["ghana", "ghanaian", "accra"],
    "Ivory Coast": ["ivory coast", "cote d'ivoire", "côte d’ivoire"],
    "Kenya": ["kenya", "kenyan", "nairobi"],
    "Libya": ["libya", "libyan", "tripoli"],
    "Madagascar": ["madagascar", "malagasy"],
    "Mali": ["mali", "malian"],
    "Morocco": ["morocco", "moroccan", "rabat", "casablanca"],
    "Mozambique": ["mozambique", "mozambican"],
    "Namibia": ["namibia", "namibian"],
    "Nigeria": ["nigeria", "nigerian", "abuja", "lagos"],
    "Rwanda": ["rwanda", "rwandan", "kigali"],
    "Senegal": ["senegal", "senegalese", "dakar"],
    "Somalia": ["somalia", "somali", "mogadishu"],
    "South Africa": ["south africa", "south african", "cape town", "johannesburg", "pretoria"],
    "Sudan": ["sudan", "sudanese", "khartoum"],
    "Tanzania": ["tanzania", "tanzanian", "dar es salaam"],
    "Tunisia": ["tunisia", "tunisian", "tunis"],
    "Uganda": ["uganda", "ugandan", "kampala"],
    "Zambia": ["zambia", "zambian"],
    "Zimbabwe": ["zimbabwe", "zimbabwean"],

    "Australia": ["australia", "australian", "canberra", "sydney", "melbourne"],
    "New Zealand": ["new zealand", "new zealander", "auckland", "wellington"],
    "Fiji": ["fiji", "fijian"],
    "Papua New Guinea": ["papua new guinea"],
    "Samoa": ["samoa", "samoan"],
    "Tonga": ["tonga", "tongan"],
}

REGION_COORDINATES = {
    # Global / regional aggregates
    "World": {"lat": 0.0, "lon": 0.0},
    "Europe": {"lat": 54.5260, "lon": 15.2551},
    "European Union": {"lat": 50.8503, "lon": 4.3517},  # Brussels
    "Africa": {"lat": 8.7832, "lon": 34.5085},
    "Asia": {"lat": 34.0479, "lon": 100.6197},
    "Middle East": {"lat": 29.2985, "lon": 42.5510},
    "North America": {"lat": 54.5260, "lon": -105.2551},
    "South America": {"lat": -8.7832, "lon": -55.4915},
    "Latin America": {"lat": -14.2350, "lon": -51.9253},
    "Oceania": {"lat": -22.7359, "lon": 140.0188},
    "Arctic": {"lat": 66.5622, "lon": 0.0},
    "Antarctica": {"lat": -82.8628, "lon": 135.0000},

    # Europe
    "Albania": {"lat": 41.1533, "lon": 20.1683},
    "Andorra": {"lat": 42.5063, "lon": 1.5218},
    "Austria": {"lat": 47.5162, "lon": 14.5501},
    "Belarus": {"lat": 53.7098, "lon": 27.9534},
    "Belgium": {"lat": 50.5039, "lon": 4.4699},
    "Bosnia and Herzegovina": {"lat": 43.9159, "lon": 17.6791},
    "Bulgaria": {"lat": 42.7339, "lon": 25.4858},
    "Croatia": {"lat": 45.1000, "lon": 15.2000},
    "Cyprus": {"lat": 35.1264, "lon": 33.4299},
    "Czech Republic": {"lat": 49.8175, "lon": 15.4730},
    "Denmark": {"lat": 56.2639, "lon": 9.5018},
    "Estonia": {"lat": 58.5953, "lon": 25.0136},
    "Finland": {"lat": 61.9241, "lon": 25.7482},
    "France": {"lat": 46.2276, "lon": 2.2137},
    "Germany": {"lat": 51.1657, "lon": 10.4515},
    "Greece": {"lat": 39.0742, "lon": 21.8243},
    "Hungary": {"lat": 47.1625, "lon": 19.5033},
    "Iceland": {"lat": 64.9631, "lon": -19.0208},
    "Ireland": {"lat": 53.4129, "lon": -8.2439},
    "Italy": {"lat": 41.8719, "lon": 12.5674},
    "Kosovo": {"lat": 42.6026, "lon": 20.9030},
    "Latvia": {"lat": 56.8796, "lon": 24.6032},
    "Liechtenstein": {"lat": 47.1660, "lon": 9.5554},
    "Lithuania": {"lat": 55.1694, "lon": 23.8813},
    "Luxembourg": {"lat": 49.8153, "lon": 6.1296},
    "Malta": {"lat": 35.9375, "lon": 14.3754},
    "Moldova": {"lat": 47.4116, "lon": 28.3699},
    "Monaco": {"lat": 43.7384, "lon": 7.4246},
    "Montenegro": {"lat": 42.7087, "lon": 19.3744},
    "Netherlands": {"lat": 52.1326, "lon": 5.2913},
    "North Macedonia": {"lat": 41.6086, "lon": 21.7453},
    "Norway": {"lat": 60.4720, "lon": 8.4689},
    "Poland": {"lat": 51.9194, "lon": 19.1451},
    "Portugal": {"lat": 39.3999, "lon": -8.2245},
    "Romania": {"lat": 45.9432, "lon": 24.9668},
    "Russia": {"lat": 61.5240, "lon": 105.3188},
    "San Marino": {"lat": 43.9424, "lon": 12.4578},
    "Serbia": {"lat": 44.0165, "lon": 21.0059},
    "Slovakia": {"lat": 48.6690, "lon": 19.6990},
    "Slovenia": {"lat": 46.1512, "lon": 14.9955},
    "Spain": {"lat": 40.4637, "lon": -3.7492},
    "Sweden": {"lat": 60.1282, "lon": 18.6435},
    "Switzerland": {"lat": 46.8182, "lon": 8.2275},
    "Turkey": {"lat": 38.9637, "lon": 35.2433},
    "Ukraine": {"lat": 48.3794, "lon": 31.1656},
    "United Kingdom": {"lat": 55.3781, "lon": -3.4360},
    "Vatican City": {"lat": 41.9029, "lon": 12.4534},

    # North America
    "United States": {"lat": 37.0902, "lon": -95.7129},
    "Canada": {"lat": 56.1304, "lon": -106.3468},
    "Mexico": {"lat": 23.6345, "lon": -102.5528},

    # Central America / Caribbean
    "Belize": {"lat": 17.1899, "lon": -88.4976},
    "Costa Rica": {"lat": 9.7489, "lon": -83.7534},
    "El Salvador": {"lat": 13.7942, "lon": -88.8965},
    "Guatemala": {"lat": 15.7835, "lon": -90.2308},
    "Honduras": {"lat": 15.2000, "lon": -86.2419},
    "Nicaragua": {"lat": 12.8654, "lon": -85.2072},
    "Panama": {"lat": 8.5380, "lon": -80.7821},
    "Cuba": {"lat": 21.5218, "lon": -77.7812},
    "Haiti": {"lat": 18.9712, "lon": -72.2852},
    "Dominican Republic": {"lat": 18.7357, "lon": -70.1627},
    "Jamaica": {"lat": 18.1096, "lon": -77.2975},
    "Bahamas": {"lat": 25.0343, "lon": -77.3963},
    "Barbados": {"lat": 13.1939, "lon": -59.5432},
    "Trinidad and Tobago": {"lat": 10.6918, "lon": -61.2225},

    # South America
    "Argentina": {"lat": -38.4161, "lon": -63.6167},
    "Bolivia": {"lat": -16.2902, "lon": -63.5887},
    "Brazil": {"lat": -14.2350, "lon": -51.9253},
    "Chile": {"lat": -35.6751, "lon": -71.5430},
    "Colombia": {"lat": 4.5709, "lon": -74.2973},
    "Ecuador": {"lat": -1.8312, "lon": -78.1834},
    "Guyana": {"lat": 4.8604, "lon": -58.9302},
    "Paraguay": {"lat": -23.4425, "lon": -58.4438},
    "Peru": {"lat": -9.1900, "lon": -75.0152},
    "Suriname": {"lat": 3.9193, "lon": -56.0278},
    "Uruguay": {"lat": -32.5228, "lon": -55.7658},
    "Venezuela": {"lat": 6.4238, "lon": -66.5897},

    # Middle East
    "Iran": {"lat": 32.4279, "lon": 53.6880},
    "Iraq": {"lat": 33.2232, "lon": 43.6793},
    "Israel": {"lat": 31.0461, "lon": 34.8516},
    "Palestine": {"lat": 31.9522, "lon": 35.2332},
    "Lebanon": {"lat": 33.8547, "lon": 35.8623},
    "Syria": {"lat": 34.8021, "lon": 38.9968},
    "Jordan": {"lat": 30.5852, "lon": 36.2384},
    "Saudi Arabia": {"lat": 23.8859, "lon": 45.0792},
    "United Arab Emirates": {"lat": 23.4241, "lon": 53.8478},
    "Qatar": {"lat": 25.3548, "lon": 51.1839},
    "Kuwait": {"lat": 29.3117, "lon": 47.4818},
    "Bahrain": {"lat": 26.0667, "lon": 50.5577},
    "Oman": {"lat": 21.4735, "lon": 55.9754},
    "Yemen": {"lat": 15.5527, "lon": 48.5164},

    # Asia
    "Afghanistan": {"lat": 33.9391, "lon": 67.7100},
    "Armenia": {"lat": 40.0691, "lon": 45.0382},
    "Azerbaijan": {"lat": 40.1431, "lon": 47.5769},
    "Bangladesh": {"lat": 23.6850, "lon": 90.3563},
    "Bhutan": {"lat": 27.5142, "lon": 90.4336},
    "Cambodia": {"lat": 12.5657, "lon": 104.9910},
    "China": {"lat": 35.8617, "lon": 104.1954},
    "Georgia": {"lat": 42.3154, "lon": 43.3569},
    "India": {"lat": 20.5937, "lon": 78.9629},
    "Indonesia": {"lat": -0.7893, "lon": 113.9213},
    "Japan": {"lat": 36.2048, "lon": 138.2529},
    "Kazakhstan": {"lat": 48.0196, "lon": 66.9237},
    "Kyrgyzstan": {"lat": 41.2044, "lon": 74.7661},
    "Laos": {"lat": 19.8563, "lon": 102.4955},
    "Malaysia": {"lat": 4.2105, "lon": 101.9758},
    "Maldives": {"lat": 3.2028, "lon": 73.2207},
    "Mongolia": {"lat": 46.8625, "lon": 103.8467},
    "Myanmar": {"lat": 21.9162, "lon": 95.9560},
    "Nepal": {"lat": 28.3949, "lon": 84.1240},
    "North Korea": {"lat": 40.3399, "lon": 127.5101},
    "Pakistan": {"lat": 30.3753, "lon": 69.3451},
    "Philippines": {"lat": 12.8797, "lon": 121.7740},
    "Singapore": {"lat": 1.3521, "lon": 103.8198},
    "South Korea": {"lat": 35.9078, "lon": 127.7669},
    "Sri Lanka": {"lat": 7.8731, "lon": 80.7718},
    "Taiwan": {"lat": 23.6978, "lon": 120.9605},
    "Tajikistan": {"lat": 38.8610, "lon": 71.2761},
    "Thailand": {"lat": 15.8700, "lon": 100.9925},
    "Turkmenistan": {"lat": 38.9697, "lon": 59.5563},
    "Uzbekistan": {"lat": 41.3775, "lon": 64.5853},
    "Vietnam": {"lat": 14.0583, "lon": 108.2772},

    # Africa
    "Algeria": {"lat": 28.0339, "lon": 1.6596},
    "Angola": {"lat": -11.2027, "lon": 17.8739},
    "Botswana": {"lat": -22.3285, "lon": 24.6849},
    "Cameroon": {"lat": 7.3697, "lon": 12.3547},
    "Democratic Republic of the Congo": {"lat": -4.0383, "lon": 21.7587},
    "Republic of the Congo": {"lat": -0.2280, "lon": 15.8277},
    "Egypt": {"lat": 26.8206, "lon": 30.8025},
    "Ethiopia": {"lat": 9.1450, "lon": 40.4897},
    "Ghana": {"lat": 7.9465, "lon": -1.0232},
    "Ivory Coast": {"lat": 7.5400, "lon": -5.5471},
    "Kenya": {"lat": -0.0236, "lon": 37.9062},
    "Libya": {"lat": 26.3351, "lon": 17.2283},
    "Madagascar": {"lat": -18.7669, "lon": 46.8691},
    "Mali": {"lat": 17.5707, "lon": -3.9962},
    "Morocco": {"lat": 31.7917, "lon": -7.0926},
    "Mozambique": {"lat": -18.6657, "lon": 35.5296},
    "Namibia": {"lat": -22.9576, "lon": 18.4904},
    "Nigeria": {"lat": 9.0820, "lon": 8.6753},
    "Rwanda": {"lat": -1.9403, "lon": 29.8739},
    "Senegal": {"lat": 14.4974, "lon": -14.4524},
    "Somalia": {"lat": 5.1521, "lon": 46.1996},
    "South Africa": {"lat": -30.5595, "lon": 22.9375},
    "Sudan": {"lat": 12.8628, "lon": 30.2176},
    "Tanzania": {"lat": -6.3690, "lon": 34.8888},
    "Tunisia": {"lat": 33.8869, "lon": 9.5375},
    "Uganda": {"lat": 1.3733, "lon": 32.2903},
    "Zambia": {"lat": -13.1339, "lon": 27.8493},
    "Zimbabwe": {"lat": -19.0154, "lon": 29.1549},

    # Oceania
    "Australia": {"lat": -25.2744, "lon": 133.7751},
    "New Zealand": {"lat": -40.9006, "lon": 174.8860},
    "Fiji": {"lat": -17.7134, "lon": 178.0650},
    "Papua New Guinea": {"lat": -6.3150, "lon": 143.9555},
    "Samoa": {"lat": -13.7590, "lon": -172.1046},
    "Tonga": {"lat": -21.1790, "lon": -175.1982},
}

BOILERPLATE_PATTERNS = [
    r"\bcontinue reading\b",
    r"\bread more\b",
    r"\bclick here\b",
    r"\bsubscribe now\b",
    r"\bsign up\b",
    r"\bfollow live\b",
    r"\blive updates\b",
    r"\bthis story has been updated\b",
    r"\badvertisement\b",
    r"\bskip advertisement\b",
    r"\bfull story\b",
    r"\bfor more information\b",
]

CUSTOM_TOPIC_STOPWORDS = {
    "continue",
    "reading",
    "read",
    "more",
    "click",
    "here",
    "subscribe",
    "sign",
    "watch",
    "video",
    "live",
    "updates",
    "breaking",
    "news",
    "article",
    "story",
    "said",
    "says",
    "say",
    "new",
    "year",
    "years",
    "like",
    "just",
    "people",
    "time",
    "day",
    "week",
    "month",
    "today",
    "latest",
    "report",
    "reports",
    "reported",
    "according",
    "online",
    "full",
    "advertisement",
    "skip",
}


def clean_text(text: str | None) -> str:
    if text is None:
        return ""

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜßéèàçÉÈÀÇ\s]", "", text)
    text = text.strip()

    return text


def clean_topic_text(text: str | None) -> str:
    if text is None:
        return ""

    if not isinstance(text, str):
        return ""

    text = text.lower()

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text)

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        token
        for token in text.split()
        if token not in CUSTOM_TOPIC_STOPWORDS and len(token) > 2
    ]

    return deduplicate_tokens(" ".join(tokens))


def normalize_for_dedup(text: str | None) -> str:
    if text is None:
        return ""

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def deduplicate_tokens(text: str | None) -> str:
    if text is None:
        return ""

    if not isinstance(text, str):
        return ""

    tokens = text.split()
    seen = set()
    deduped_tokens = []

    for token in tokens:
        if token in seen:
            continue

        seen.add(token)
        deduped_tokens.append(token)

    return " ".join(deduped_tokens)


def remove_duplicate_articles(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    before = len(news_df)

    news_df["headline_dedup_key"] = news_df["headline"].apply(normalize_for_dedup)

    if "url" in news_df.columns:
        news_df["url_dedup_key"] = news_df["url"].fillna("").astype(str).str.strip()
    else:
        news_df["url_dedup_key"] = ""

    # First remove exact URL duplicates where URL exists.
    has_url = news_df["url_dedup_key"].str.len() > 0
    with_url = news_df.loc[has_url].drop_duplicates(
        subset=["url_dedup_key"],
        keep="first",
    )
    without_url = news_df.loc[~has_url]

    news_df = pd.concat([with_url, without_url], ignore_index=True)

    # Then remove duplicate / near-identical headlines.
    news_df = news_df.drop_duplicates(
        subset=["headline_dedup_key"],
        keep="first",
    )

    news_df = news_df.drop(
        columns=["headline_dedup_key", "url_dedup_key"],
        errors="ignore",
    )

    after = len(news_df)
    print(f"Removed duplicate articles: {before - after}")

    return news_df


def clean_news_data(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    news_df = news_df.dropna(subset=["headline"])
    news_df["description"] = news_df["description"].fillna("")

    news_df = remove_duplicate_articles(news_df)

    news_df["headline_clean"] = news_df["headline"].apply(clean_text)
    news_df["description_clean"] = news_df["description"].apply(clean_text)

    return news_df


def add_text_features(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    news_df["headline_length"] = news_df["headline_clean"].str.len()
    news_df["headline_word_count"] = news_df["headline_clean"].str.split().str.len()

    news_df["description_length"] = news_df["description_clean"].str.len()
    news_df["description_word_count"] = news_df["description_clean"].str.split().str.len()

    return news_df


def add_time_features(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    news_df["published_at_parsed"] = pd.to_datetime(
        news_df["published_at"],
        errors="coerce",
        utc=True,
    )

    news_df["publication_hour"] = news_df["published_at_parsed"].dt.hour
    news_df["publication_day_of_week"] = news_df["published_at_parsed"].dt.day_name()

    return news_df


def combine_text_for_region(headline: str | None, description: str | None) -> str:
    text_parts = []

    if isinstance(headline, str):
        text_parts.append(headline)

    if isinstance(description, str):
        text_parts.append(description)

    combined_text = " ".join(text_parts)
    combined_text = combined_text.lower()

    return combined_text


def contains_keyword(text: str, keyword: str) -> bool:
    text_with_spaces = f" {text} "
    keyword_with_spaces = f" {keyword} "

    return keyword_with_spaces in text_with_spaces


def extract_region(headline: str | None, description: str | None) -> str:
    text = combine_text_for_region(headline, description)

    for region, keywords in REGION_KEYWORDS.items():
        for keyword in keywords:
            if contains_keyword(text, keyword):
                return region

    return "World"


def add_region_feature(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    regions = []

    for row in news_df.itertuples(index=False):
        region = extract_region(
            headline=getattr(row, "headline", None),
            description=getattr(row, "description", None),
        )
        regions.append(region)

    def get_lat(region: str) -> float:
        return REGION_COORDINATES.get(
            region,
            REGION_COORDINATES["World"],
        )["lat"]

    def get_lon(region: str) -> float:
        return REGION_COORDINATES.get(
            region,
            REGION_COORDINATES["World"],
        )["lon"]

    news_df["region"] = regions
    news_df["lat"] = news_df["region"].apply(get_lat)
    news_df["lon"] = news_df["region"].apply(get_lon)

    return news_df


def add_topic_text(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    combined_text = (
        news_df["headline_clean"].fillna("")
        + " "
        + news_df["description_clean"].fillna("")
    )

    news_df["topic_text"] = combined_text.apply(clean_topic_text)

    return news_df


def add_features(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    news_df = add_text_features(news_df)
    news_df = add_time_features(news_df)
    news_df = add_region_feature(news_df)
    news_df = add_topic_text(news_df)

    return news_df


def get_raw_file_for_date(run_date: str, raw_dir: str = "data/raw/rss/live") -> Path:
    raw_file = Path(raw_dir) / f"raw_news_{run_date}.csv"

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file}")

    return raw_file


def preprocess_news(input_path: str | Path) -> pd.DataFrame:
    news_df = pd.read_csv(input_path)

    news_df = clean_news_data(news_df)
    news_df = add_features(news_df)

    return news_df


def save_processed_news(
    news_df: pd.DataFrame,
    output_path: str | Path,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    news_df.to_csv(output_path, index=False, encoding="utf-8")

    return str(output_path)


def get_default_output_path(run_date: str) -> Path:
    output_path = Path("data/processed/live") / f"processed_news_{run_date}.csv"

    return output_path


def get_api_output_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    file_name = input_path.name

    processed_file_name = file_name.replace("raw_", "processed_")
    output_path = Path("data/processed/api") / processed_file_name

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-date",
        required=False,
        default=None,
        help="Pipeline run date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--input-path",
        required=False,
        default=None,
        help="Optional raw input CSV path. Use this for API backfill files.",
    )

    parser.add_argument(
        "--output-path",
        required=False,
        default=None,
        help="Optional processed output CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_path:
        raw_file = Path(args.input_path)

        if not raw_file.exists():
            raise FileNotFoundError(f"Input file not found: {raw_file}")

        if args.output_path:
            output_path = Path(args.output_path)
        else:
            output_path = get_api_output_path(raw_file)

    else:
        if args.run_date:
            run_date = args.run_date
        else:
            run_date = datetime.now().strftime("%Y-%m-%d")

        raw_file = get_raw_file_for_date(run_date)
        output_path = get_default_output_path(run_date)

    print(f"Using raw file: {raw_file}")

    processed_df = preprocess_news(raw_file)

    print(f"Processed articles: {len(processed_df)}")

    saved_path = save_processed_news(
        news_df=processed_df,
        output_path=output_path,
    )

    print(f"Saved processed news to: {saved_path}")


if __name__ == "__main__":
    main()