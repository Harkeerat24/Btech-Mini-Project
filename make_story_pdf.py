"""Generate a sample short story PDF for the GraphRAG demo."""
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pathlib import Path

STORY_TITLE = "The Last Lighthouse"

STORY_LINES = [
    "",
    "Old Thomas had kept the lighthouse for forty years. Every night, he lit",
    "the great lamp and watched its beam sweep across the dark water. Ships",
    "came and went safely because of his unwavering diligence.",
    "",
    "One stormy evening, a young vessel drifted dangerously close to the",
    "rocky shore. Thomas climbed the tower, ignoring his aching knees, and",
    "manually increased the lamp's intensity to its maximum.",
    "",
    "The captain spotted the beam just in time and steered his crew away",
    "from certain disaster. The next morning, the grateful captain visited",
    "Thomas at the base of the tower.",
    "",
    '"You saved my crew,\" he said, shaking the old keeper\'s weathered hand.',
    "",
    'Thomas simply smiled and replied, "That is what lighthouses are for."',
]

out_path = Path(__file__).parent / "sample_data" / "sample_story.pdf"
out_path.parent.mkdir(exist_ok=True)

c = canvas.Canvas(str(out_path), pagesize=A4)
c.setFont("Helvetica-Bold", 18)
c.drawString(72, 760, STORY_TITLE)
c.setFont("Helvetica", 12)
y = 720
for line in STORY_LINES:
    if y < 80:
        c.showPage()
        y = 760
        c.setFont("Helvetica", 12)
    c.drawString(72, y, line)
    y -= 20
c.save()
print(f"PDF saved -> {out_path}")
