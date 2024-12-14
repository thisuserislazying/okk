import numpy as np
from policy import Policy

class Policy2353201(Policy):
    def __init__(self, policy_id=1):
        pass

    def get_action(self, observation, info):
        # Sort products by area (width * height) in descending order
        list_prods = sorted(
            observation["products"], 
            key=lambda prod: (
                -(prod["size"][0] * prod["size"][1]),  # Largest area first
                -min(prod["size"]),                    # Larger minimum dimension
                max(prod["size"])                      # Smaller maximum dimension
            )
        )

        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue

            prod_size = prod["size"]
            rotated_size = prod_size[::-1]

            # Iterate through all stocks to find the first fit
            for idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                # Check both orientations of the product
                for size in (prod_size, rotated_size):
                    prod_w, prod_h = size
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos = self._find_placement(stock, prod_w, prod_h)
                        if pos:  # First valid position found
                            return {"stock_idx": idx, "size": size, "position": pos}
        
        # Return failure if no valid placement is found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_placement(self, stock, prod_w, prod_h):
        stock_w, stock_h = self._get_stock_size_(stock)
        best_position = None
        best_score = float('inf')
    
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                    score = (stock_w - (x + prod_w)) + (stock_h - (y + prod_h)) * 0.5
                
                    if score < best_score:
                        best_score = score
                        best_position = (x, y)
    
        return best_position

    def _get_stock_efficiency_(self, stock):
        stock_w, stock_h = self._get_stock_size_(stock)
    
        # Count used and free spaces
        used_spaces = np.sum(stock != -1)
        total_spaces = stock_w * stock_h
    
        # Calculate fragmentation
        free_spaces = total_spaces - used_spaces
        fragmentation_score = free_spaces / total_spaces
    
        # Consider aspect ratio
        aspect_ratio = max(stock_w / stock_h, stock_h / stock_w)
    
        # Composite efficiency score
        efficiency_score = (
            fragmentation_score * 0.6 +  # Lower fragmentation is better
            (1 / aspect_ratio) * 0.4     # More square-like is better
        )
    
        return efficiency_score