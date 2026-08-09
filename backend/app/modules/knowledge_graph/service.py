import networkx as nx

# Global in-memory graph (production mein ye database-backed hoga, abhi demo ke liye in-memory)
graph = nx.Graph()


def add_verification_event(user_id, document_id, device_id, ip_address, trust_score, risk_level):
    """
    Har verification attempt ke baad graph mein nodes aur relationships add karta hai.
    """
    # Nodes add karo (agar already exist karte hain to update ho jayenge)
    graph.add_node(user_id, type="user")
    graph.add_node(document_id, type="document")
    graph.add_node(device_id, type="device")
    graph.add_node(ip_address, type="ip_address")

    # Relationships (edges) add karo
    graph.add_edge(user_id, document_id, relation="submitted_document")
    graph.add_edge(user_id, device_id, relation="used_device")
    graph.add_edge(user_id, ip_address, relation="connected_from")

    # Verification metadata store karo node attribute ke tor pe
    graph.nodes[user_id]["last_trust_score"] = trust_score
    graph.nodes[user_id]["last_risk_level"] = risk_level

    return {"status": "event_added", "total_nodes": graph.number_of_nodes(), "total_edges": graph.number_of_edges()}


def detect_shared_document(document_id):
    """
    Check karta hai kya same document multiple users ne use kiya (fraud signal).
    """
    if document_id not in graph:
        return {"document_id": document_id, "linked_users": [], "risk": "unknown"}

    linked_users = [
        n for n in graph.neighbors(document_id)
        if graph.nodes[n].get("type") == "user"
    ]

    risk = "HIGH — shared document detected" if len(linked_users) > 1 else "LOW"

    return {"document_id": document_id, "linked_users": linked_users, "risk": risk}


def detect_shared_device(device_id):
    """
    Check karta hai kya same device se multiple users verify ho rahe hain.
    """
    if device_id not in graph:
        return {"device_id": device_id, "linked_users": [], "risk": "unknown"}

    linked_users = [
        n for n in graph.neighbors(device_id)
        if graph.nodes[n].get("type") == "user"
    ]

    risk = "HIGH — multiple identities from same device" if len(linked_users) > 1 else "LOW"

    return {"device_id": device_id, "linked_users": linked_users, "risk": risk}


def get_user_network(user_id):
    """
    Ek user ke saare connections (documents, devices, IPs) return karta hai.
    """
    if user_id not in graph:
        return {"user_id": user_id, "connections": []}

    connections = []
    for neighbor in graph.neighbors(user_id):
        connections.append({
            "node": neighbor,
            "type": graph.nodes[neighbor].get("type"),
            "relation": graph.edges[user_id, neighbor].get("relation")
        })

    return {
        "user_id": user_id,
        "trust_score": graph.nodes[user_id].get("last_trust_score"),
        "risk_level": graph.nodes[user_id].get("last_risk_level"),
        "connections": connections
    }


def get_graph_stats():
    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "node_types": {
            "users": len([n for n, d in graph.nodes(data=True) if d.get("type") == "user"]),
            "documents": len([n for n, d in graph.nodes(data=True) if d.get("type") == "document"]),
            "devices": len([n for n, d in graph.nodes(data=True) if d.get("type") == "device"]),
            "ip_addresses": len([n for n, d in graph.nodes(data=True) if d.get("type") == "ip_address"]),
        }
    }