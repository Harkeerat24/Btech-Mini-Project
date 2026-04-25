"""
generate_sample_pdf.py
Generates a rich sample PDF about Computer Networks & Routing Protocols.
Designed to produce many NER entities and triplets for the GraphRAG demo.

Run: python generate_sample_pdf.py
Output: sample_data/computer_networks.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT

OUT_DIR = Path(__file__).parent / "sample_data"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "computer_networks.pdf"

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
style_title    = ParagraphStyle("Title2",    parent=styles["Title"],   fontSize=22, spaceAfter=12, alignment=TA_CENTER)
style_h1       = ParagraphStyle("H1",        parent=styles["Heading1"], fontSize=16, spaceBefore=18, spaceAfter=8)
style_h2       = ParagraphStyle("H2",        parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
style_body     = ParagraphStyle("Body",      parent=styles["Normal"],   fontSize=10.5, leading=16, alignment=TA_JUSTIFY, spaceAfter=8)
style_caption  = ParagraphStyle("Caption",   parent=styles["Normal"],   fontSize=9,   textColor=colors.grey, alignment=TA_CENTER)

# ── Content ───────────────────────────────────────────────────────────────────
CONTENT = [
    # ── Title Page ──────────────────────────────────────────────────────────
    Paragraph("Computer Networks &amp; Routing Protocols", style_title),
    Paragraph("A Comprehensive Technical Reference", style_caption),
    Spacer(1, 0.5*cm),
    Paragraph(
        "This document covers the fundamental concepts of computer networking, "
        "with a focus on routing protocols, graph algorithms, and network architectures. "
        "It serves as a reference for understanding how modern internet infrastructure operates.",
        style_body
    ),
    Spacer(1, 0.5*cm),

    # ── Chapter 1 ────────────────────────────────────────────────────────────
    Paragraph("Chapter 1: Introduction to Computer Networks", style_h1),
    Paragraph("1.1 What is a Computer Network?", style_h2),
    Paragraph(
        "A computer network is a set of interconnected devices that communicate with each other "
        "using standardized protocols. The Internet is the largest computer network in the world, "
        "connecting billions of devices across every continent. Vint Cerf and Bob Kahn designed "
        "the TCP/IP protocol suite, which became the foundation of the modern Internet. "
        "The OSI model, developed by the International Organization for Standardization (ISO), "
        "defines seven layers of network communication: Physical, Data Link, Network, Transport, "
        "Session, Presentation, and Application.",
        style_body
    ),
    Paragraph("1.2 Network Topologies", style_h2),
    Paragraph(
        "Network topology refers to the arrangement of nodes and links in a network. "
        "A bus topology connects all devices to a single cable. A ring topology forms a circular "
        "chain of devices. A star topology uses a central hub or switch to connect all nodes. "
        "A mesh topology connects every device to every other device, providing maximum redundancy. "
        "The Internet uses a hybrid topology that combines elements of all these types.",
        style_body
    ),
    Paragraph("1.3 The TCP/IP Protocol Suite", style_h2),
    Paragraph(
        "The TCP/IP model uses four layers: the Network Access layer handles physical transmission, "
        "the Internet layer uses the IP protocol to route packets, the Transport layer uses TCP "
        "for reliable delivery and UDP for fast, connectionless communication, and the Application "
        "layer includes protocols like HTTP, DNS, DHCP, and FTP. TCP uses a three-way handshake "
        "to establish a connection. UDP does not establish a connection before transmitting data. "
        "The IP protocol assigns a unique address to each device on the network.",
        style_body
    ),

    # ── Chapter 2 ────────────────────────────────────────────────────────────
    Paragraph("Chapter 2: Routing and Routing Protocols", style_h1),
    Paragraph("2.1 What is Routing?", style_h2),
    Paragraph(
        "Routing is the process of selecting a path across one or more networks to send data "
        "from a source to a destination. A router is a network device that forwards data packets "
        "between computer networks. Routers use routing tables to determine the best path for "
        "each packet. Static routing uses manually configured routes that do not change. "
        "Dynamic routing uses protocols that automatically discover and update routes based on "
        "the current network topology.",
        style_body
    ),
    Paragraph("2.2 OSPF — Open Shortest Path First", style_h2),
    Paragraph(
        "OSPF is a link-state routing protocol that uses Dijkstra's algorithm to compute the "
        "shortest path tree. OSPF was developed by the IETF and is defined in RFC 2328. "
        "OSPF uses Hello packets to discover and maintain neighbor relationships. Routers that "
        "share a common network segment form an adjacency. OSPF floods Link State Advertisements "
        "(LSAs) throughout the network to ensure every router has an identical copy of the "
        "link-state database. OSPF divides a network into areas to reduce the volume of routing "
        "traffic. Area 0, also called the backbone area, connects all other areas. "
        "When a link failure occurs, OSPF detects the failure through the expiration of the "
        "Dead Interval timer. The router then floods a new LSA and every router re-runs "
        "Dijkstra's algorithm to recalculate the shortest path tree. OSPF supports VLSM and CIDR "
        "and is considered a classless routing protocol.",
        style_body
    ),
    Paragraph("2.3 BGP — Border Gateway Protocol", style_h2),
    Paragraph(
        "BGP is the routing protocol that powers the Internet. BGP is a path-vector routing "
        "protocol that makes routing decisions based on paths, network policies, and rule-sets. "
        "BGP uses TCP port 179 to establish sessions between routers. An Autonomous System (AS) "
        "is a collection of IP networks under the control of a single organization. BGP routers "
        "exchange Network Layer Reachability Information (NLRI) to advertise reachable prefixes. "
        "iBGP runs within the same AS, while eBGP runs between different ASes. "
        "The BGP route selection process uses the AS_PATH attribute to detect and prevent routing "
        "loops. The LOCAL_PREF attribute determines the preferred exit point from an AS. "
        "BGP uses the MED attribute to influence how traffic enters an AS from an external peer.",
        style_body
    ),
    Paragraph("2.4 RIP — Routing Information Protocol", style_h2),
    Paragraph(
        "RIP is one of the oldest distance-vector routing protocols. RIP uses hop count as its "
        "metric, with a maximum hop count of 15. A hop count of 16 represents an unreachable "
        "network. RIP routers broadcast their entire routing table every 30 seconds. "
        "RIPv2 added support for classless routing and multicast updates. "
        "RIP uses the Bellman-Ford algorithm to calculate the best path. "
        "RIP suffers from the count-to-infinity problem, which is mitigated by split horizon "
        "and route poisoning techniques.",
        style_body
    ),
    Paragraph("2.5 EIGRP — Enhanced Interior Gateway Routing Protocol", style_h2),
    Paragraph(
        "EIGRP is an advanced distance-vector routing protocol developed by Cisco. "
        "EIGRP uses the Diffusing Update Algorithm (DUAL) to guarantee loop-free paths. "
        "EIGRP uses a composite metric based on bandwidth and delay. "
        "EIGRP maintains a topology table that stores all routes learned from neighbors. "
        "The successor is the primary route to a destination. The feasible successor is a "
        "backup route that satisfies the feasibility condition. EIGRP sends partial updates only "
        "when the network topology changes, unlike RIP which sends full updates periodically.",
        style_body
    ),

    # ── Chapter 3 ────────────────────────────────────────────────────────────
    Paragraph("Chapter 3: Graph Algorithms in Networking", style_h1),
    Paragraph("3.1 Dijkstra's Algorithm", style_h2),
    Paragraph(
        "Dijkstra's algorithm, invented by Edsger Dijkstra in 1956, computes the shortest path "
        "from a single source to all other vertices in a weighted graph with non-negative edge weights. "
        "OSPF uses Dijkstra's algorithm to build the shortest path tree from each router's perspective. "
        "The algorithm maintains a priority queue of unvisited vertices sorted by their current "
        "shortest distance. The time complexity of Dijkstra's algorithm is O((V + E) log V) when "
        "implemented with a binary heap. The algorithm produces the shortest path tree (SPT), "
        "which OSPF stores in its routing table.",
        style_body
    ),
    Paragraph("3.2 Bellman-Ford Algorithm", style_h2),
    Paragraph(
        "The Bellman-Ford algorithm computes the shortest path from a single source to all other "
        "vertices in a weighted graph. Unlike Dijkstra's algorithm, Bellman-Ford can handle negative "
        "edge weights. RIP uses the Bellman-Ford algorithm for its distance-vector computations. "
        "Bellman-Ford has a time complexity of O(V * E). The algorithm can detect negative weight "
        "cycles, which indicate a routing loop in network terms. BGP prevents routing loops using "
        "the AS_PATH attribute rather than relying on Bellman-Ford.",
        style_body
    ),
    Paragraph("3.3 Spanning Tree Protocol", style_h2),
    Paragraph(
        "The Spanning Tree Protocol (STP), defined in IEEE 802.1D, prevents Layer 2 loops in "
        "Ethernet networks. STP uses a distributed algorithm to elect a root bridge. "
        "All switches elect the bridge with the lowest Bridge ID as the root bridge. "
        "STP then constructs a spanning tree rooted at the root bridge, disabling redundant links. "
        "Rapid Spanning Tree Protocol (RSTP), defined in IEEE 802.1w, converges much faster than STP. "
        "Multiple Spanning Tree Protocol (MSTP), defined in IEEE 802.1s, allows multiple VLANs "
        "to be mapped to different spanning tree instances.",
        style_body
    ),

    PageBreak(),

    # ── Chapter 4 ────────────────────────────────────────────────────────────
    Paragraph("Chapter 4: Network Security", style_h1),
    Paragraph("4.1 Firewalls and Access Control Lists", style_h2),
    Paragraph(
        "A firewall is a network security device that monitors and controls incoming and outgoing "
        "network traffic based on predetermined security rules. Cisco developed the first "
        "commercial firewall in 1992. A stateful firewall tracks the state of active connections. "
        "A stateless firewall filters packets based on static rules without tracking connection state. "
        "Access Control Lists (ACLs) are used by routers and switches to filter traffic. "
        "An ACL uses permit and deny statements to control which traffic is allowed to pass.",
        style_body
    ),
    Paragraph("4.2 VPNs and Encryption", style_h2),
    Paragraph(
        "A Virtual Private Network (VPN) creates an encrypted tunnel over a public network. "
        "IPSec is a protocol suite that provides authentication and encryption at the IP layer. "
        "SSL/TLS VPNs use the Transport Layer Security protocol to secure communications. "
        "MPLS VPNs use Multi-Protocol Label Switching to create private network segments. "
        "WireGuard is a modern VPN protocol that uses state-of-the-art cryptography. "
        "OpenVPN is an open-source VPN solution that uses OpenSSL for encryption.",
        style_body
    ),
    Paragraph("4.3 DNS Security", style_h2),
    Paragraph(
        "The Domain Name System (DNS) translates human-readable domain names to IP addresses. "
        "DNS uses a hierarchical distributed database. The root nameservers are the top of the "
        "DNS hierarchy. DNSSEC adds cryptographic signatures to DNS records to prevent spoofing. "
        "DNS over HTTPS (DoH) encrypts DNS queries to protect user privacy. "
        "DNS amplification attacks exploit open DNS resolvers to perform distributed denial-of-service attacks.",
        style_body
    ),

    # ── Chapter 5 ────────────────────────────────────────────────────────────
    Paragraph("Chapter 5: Software-Defined Networking (SDN)", style_h1),
    Paragraph("5.1 SDN Architecture", style_h2),
    Paragraph(
        "Software-Defined Networking (SDN) separates the control plane from the data plane in "
        "network devices. The SDN controller is a centralized software application that manages "
        "the entire network. OpenFlow is the most widely used protocol for communication between "
        "the SDN controller and network devices. Nick McKeown and Scott Shenker pioneered the "
        "concept of SDN at Stanford University. "
        "The data plane handles packet forwarding based on rules installed by the control plane. "
        "SDN enables network programmability and automation through northbound APIs.",
        style_body
    ),
    Paragraph("5.2 Network Function Virtualization", style_h2),
    Paragraph(
        "Network Function Virtualization (NFV) replaces dedicated hardware network appliances "
        "with software running on commodity servers. NFV uses virtual machines or containers to "
        "host network functions such as firewalls, load balancers, and routers. "
        "ETSI defined the NFV architecture framework. "
        "Kubernetes is commonly used to orchestrate containerized network functions. "
        "Docker containers provide lightweight isolation for virtualized network functions. "
        "NFV and SDN are complementary technologies that together enable cloud-native networking.",
        style_body
    ),

    # ── Summary Table ────────────────────────────────────────────────────────
    Paragraph("Summary: Routing Protocol Comparison", style_h1),
    Spacer(1, 0.3*cm),
]

# Table data
table_data = [
    ["Protocol", "Type",        "Algorithm",    "Metric",        "Convergence"],
    ["OSPF",     "Link-State",  "Dijkstra",     "Cost",          "Fast"],
    ["BGP",      "Path-Vector", "Policy-based", "AS Path / MED", "Slow"],
    ["RIP",      "Distance-Vec","Bellman-Ford", "Hop Count",     "Very Slow"],
    ["EIGRP",    "Adv. D-Vec",  "DUAL",         "BW + Delay",    "Fast"],
    ["IS-IS",    "Link-State",  "Dijkstra",     "Cost",          "Fast"],
]

table = Table(table_data, colWidths=[3*cm, 3.5*cm, 3.5*cm, 4*cm, 3*cm])
table.setStyle(TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#4A4A8A")),
    ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
    ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, 0),  10),
    ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F0F0F8"), colors.white]),
    ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#AAAACC")),
    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE",    (0, 1), (-1, -1), 9.5),
    ("ROWHEIGHT",   (0, 0), (-1, -1), 0.7*cm),
]))
CONTENT.append(table)
CONTENT.append(Spacer(1, 0.5*cm))
CONTENT.append(Paragraph(
    "Table 1: Comparison of major IP routing protocols used in the Internet.",
    style_caption
))

# ── Build PDF ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    str(OUT_PATH),
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm,  bottomMargin=2.5*cm,
)
doc.build(CONTENT)
print(f"[OK] PDF generated: {OUT_PATH}")
print(f"   Size: {OUT_PATH.stat().st_size / 1024:.1f} KB")
print(f"   Pages: ~6 pages")
print(f"   Topics: OSPF, BGP, RIP, EIGRP, TCP/IP, DNS, SDN, Dijkstra, Bellman-Ford")
