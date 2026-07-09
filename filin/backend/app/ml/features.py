from app.core.schemas import NetworkEvent


def extract_features(event: NetworkEvent) -> dict[str, float]:
    features = dict(event.features)
    if event.duration is not None:
        features.setdefault("duration", event.duration)
    if event.bytes_in is not None:
        features.setdefault("bytes_in", float(event.bytes_in))
    if event.bytes_out is not None:
        features.setdefault("bytes_out", float(event.bytes_out))
    if event.packets is not None:
        features.setdefault("packets", float(event.packets))
    if event.destination_port is not None:
        features.setdefault("destination_port", float(event.destination_port))
    return features
