**Dependencies:**

1. pip install -r dependencies.txt
2. $env:OPENAI_API_KEY="your-api-key"
3. List of variables that are customised in the IM_pro function: 
    ->r_thresh_val = threshold value for creating bin img of red block.
    -> g_thresh_val= threshold value for creating bin img of green block.
    -> b_thresh_val= threshold value for creating bin img of blue block.
     These are found in the thresh_vals list
     -> area_min and area_max: parameters for detecting if an identified blob is a block or not.