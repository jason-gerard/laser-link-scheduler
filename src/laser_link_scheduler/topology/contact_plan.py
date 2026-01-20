import csv
import json
import os
from dataclasses import dataclass

from constants import SOURCES_ROOT
from utils import get_experiment_file, FileType


@dataclass(frozen=True)
class Contact:
    tx_node: str  # Transmitting node, from node
    rx_node: str  # Receiving node, to node
    start_time: int
    end_time: int
    bit_rate: int  # bits per second
    range: float  # distance between the nodes in light-seconds

    tx_x: float
    tx_y: float
    tx_z: float

    rx_x: float
    rx_y: float
    rx_z: float


@dataclass
class ContactPlan:
    contacts: list[Contact]


class IONContactPlanParser:
    TIMESTAMP_PREFIX = "+"

    def read(self, experiment_name: str) -> ContactPlan:
        contacts = []

        path = get_experiment_file(experiment_name, FileType.CONTACT_PLAN)
        with open(path, "r") as f:
            lines = f.read()
            # We assume here that the contact plan file has a section of contact commands followed by a blank line then
            # a section of range commands, followed by another blank line then the azimuth, elevation, range section
            # sections which contains the positioning data for each sat at the start of each contact
            contact_commands_str, range_commands_str, aer_commands_str, _ = lines.split(
                "\n\n"
            )
            contact_commands = [
                command.split(" ") for command in contact_commands_str.split("\n")
            ]
            range_commands = [
                command.split(" ") for command in range_commands_str.split("\n")
            ]
            aer_commands = [
                command.split(" ") for command in aer_commands_str.split("\n")
            ]

            # Iterate through the contact commands and range commands in order
            for contact_command, range_command, aer_command in zip(
                contact_commands, range_commands, aer_commands
            ):
                ion_start_time, ion_end_time, tx_node, rx_node, bit_rate = (
                    contact_command[2:]
                )
                range_in_light_seconds = range_command[-1]

                tx_x, tx_y, tx_z = aer_command[6:9]
                rx_x, rx_y, rx_z = aer_command[9:12]

                contact = Contact(
                    tx_node=tx_node,
                    rx_node=rx_node,
                    start_time=int(ion_start_time[1:]),
                    end_time=int(ion_end_time[1:]),
                    bit_rate=int(bit_rate),
                    range=float(range_in_light_seconds),
                    tx_x=tx_x,
                    tx_y=tx_y,
                    tx_z=tx_z,
                    rx_x=rx_x,
                    rx_y=rx_y,
                    rx_z=rx_z,
                )

                contacts.append(contact)

        return ContactPlan(contacts)

    def write(
        self, experiment_name: str, contact_plan: ContactPlan, file_type: FileType
    ):
        contact_rows = []
        range_rows = []

        for contact in contact_plan.contacts:
            ion_start_time = (
                f"{IONContactPlanParser.TIMESTAMP_PREFIX}{contact.start_time}"
            )
            ion_end_time = f"{IONContactPlanParser.TIMESTAMP_PREFIX}{contact.end_time}"

            contact_rows.append(
                [
                    "a",
                    "contact",
                    ion_start_time,
                    ion_end_time,
                    contact.tx_node,
                    contact.rx_node,
                    contact.bit_rate,
                ]
            )

            range_rows.append(
                [
                    "a",
                    "range",
                    ion_start_time,
                    ion_end_time,
                    contact.tx_node,
                    contact.rx_node,
                    contact.range,
                ]
            )

        path = get_experiment_file(experiment_name, file_type)
        with open(path, "w") as f:
            writer = csv.writer(f, delimiter=" ", lineterminator="\n")
            writer.writerows(contact_rows)
            writer.writerow("")
            writer.writerows(range_rows)


class IPNDContactPlanParser:
    def write(self, experiment_name: str, contact_plan: ContactPlan):
        contact_plan_json = {"ContactPlan": []}

        initial_start_time = 725803264.184
        for contact in contact_plan.contacts:
            contact_json = {
                "SourceID": int(contact.tx_node),
                "DestinationID": int(contact.rx_node),
                "StartTime": initial_start_time + contact.start_time,
                "EndTime": initial_start_time + contact.end_time,
                "Duration": float(contact.end_time - contact.start_time),
                "Color": [],
            }
            contact_plan_json["ContactPlan"].append(contact_json)

        path = os.path.join(SOURCES_ROOT, experiment_name, "contactPlan.json")
        with open(path, "w") as f:
            json.dump(contact_plan_json, f, indent=4)
