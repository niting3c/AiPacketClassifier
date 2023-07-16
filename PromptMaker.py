def generate_first_prompt(packet_count):
    """
    221 tokens
    Generates the initial prompt for analyzing packets.

    Args:
        packet_count (int): The number of packets to analyze.

    Returns:
        str: The generated prompt.
    """
    return """
        You are an advanced AI that detects malicious requests by parsing the various payloads of the protocols.
        Please analyze the {0} packets payload provided in the follow-up prompts 
        and determine if it is malicious or not. 
        Packet Payload if bigger, will be split into chunks and sent for analysis.
        Note that an empty payload is not considered malicious.
        As an AI classifier specialized in detecting malicious activity or network attacks, 
        you should carefully examine the payload and follow a step-by-step analysis.
        If even One of the packet is malicious , mark the whole pcap file as malicious.
        The prompt for each packet will be provided after this instruction.
    """.format(packet_count)


def generate_prompt(protocol, payload, index):
    """
    minimum 16 tokens
    Generates a prompt for analyzing a packet with the given protocol and payload.

    Args:
        protocol (str): The protocol of the packet.
        payload (str): The payload of the packet.

    Returns:
        str: The generated prompt.
    """
    return """
    Packet: {2}
    Protocol: {0}
    Payload: {1}
    """.format(protocol, payload, index)


def generate_part_prompt(protocol, payload, count, total, index):
    """
    50 tokens minimum
    Generates a prompt for analyzing a part of a packet's payload.

    Args:
        protocol (str): The protocol of the packet.
        payload (str): The payload of the packet part.
        count (int): The part count of the payload.
        total (int): The total number of parts.

    Returns:
        str: The generated prompt.
    """
    return """
    As the payload is large, we will split the payload into {3} parts.
    Here is part {2} of the payload,Do not provide explanation, Only respond if its Malicious or Not-Malicious.:
    Protocol: {0}
    Payload: {1}
    Packet: {4}
    """.format(protocol, payload, count, total, index)


def generate_part_prompt_final():
    """
    Generates the final prompt for analyzing the parts of a packet's payload.
    This needs to be called once all the parts have been sent , to conclude the request
    Returns:
        str: The generated prompt.
    """
    return """
    All the parts of the payload in the packet are provided, please analyze them as a whole
    and categorize whether they are malicious or not. Do not provide explanation, Only respond if its Malicious or Not-Malicious.
    """


def generate_text_chat_last_prompt():
    return """
    Based on all the packets sent above , is the packet capture malicious or an attack on the server/network ?
    Provide response either `Malicious` or `Not-Malicious` 
    """
