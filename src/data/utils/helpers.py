def _normalize_annotations(annotations: dict) -> list[dict]:
    """
    Transform column format annotations to row format.
    """
    if "labels" not in annotations or not annotations["labels"]:
        return []
    
    labels = annotations["labels"]
    # NOTE: already in row format
    if isinstance(labels, list):
        return labels
    
    # NOTE: in column format
    if isinstance(labels, dict):
        # NOTE: we transpose
        keys = labels.keys()
        values_per_key = labels.values()
        num_labels = len(next(iter(values_per_key), []))
        
        reformatted_labels = []
        for i in range(num_labels):
            # print(i, keys, labels)
            reformatted_labels.append({key: labels[key][i] for key in keys})
        return reformatted_labels
        
    return []