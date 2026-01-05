def retention_decision(churn_probability):
    """
    Converts churn probability into a business action
    """
    if churn_probability >= 0.80:
        return "High-priority retention: call + discount"
    elif churn_probability >= 0.50:
        return "Medium-priority retention: email campaign"
    else:
        return "No action required"
