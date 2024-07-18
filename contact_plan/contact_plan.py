from dataclasses import dataclass


@dataclass
class Contact:
    # Transmitting node, from node
    tx_node: str
    # Receiving node, to node
    rx_node: str
    start_time: int
    end_time: int
    # The context attribute can be any additional information associated with the contact. For example this could
    # include the range or distance between the contacts, data rate, BER, etc
    context: dict


@dataclass
class ContactPlan:
    contacts: list[Contact]
