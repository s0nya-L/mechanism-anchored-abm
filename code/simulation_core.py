from openai import OpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()
temperature = 0.7
top_p = 1.0
rounds = 10
client = OpenAI(api_key=os.getenv('API_KEY'), base_url="https://api.siliconflow.cn/")

"""
_________________________________________________________________________________________
variables

"""

PRODUCT_NAME = "Insulin Analog"
MARKET_VALUE = 120 #TBD
INITIAL_STOCK = 10000
PRODUCTION_COST = 40 #TBD
DECAY_RATE = 0.95 #TBC


"""self  = r1, s1"""

"""
_________________________________________________________________________________________
classes

"""

class supplier:
    def __init__(self, name, group, suppliers):
        self.name = name
        self.grp = group
        self.situation = [] #negotiation history, [number of other party][0] = other party's name. [oth][1][0] = negotiation history.. not sure why a I nested anothe rlist into this? :)
        self.stock = INITIAL_STOCK 
        self.decisions = [] #decision whether or not to continue, true = continue, false = dont continue, index corresponds to negotiation/number of other party, [number of other party][0=self of other party, 1=boolean]
        self.deals = [] #whether a deal has been reached, boolean, index corresponds to negotiation/number of other party, [number of other party][0=self of other party, 1=boolean]
        self.history = "" #temporary store of own dialogue
        self.spoilt = 0  # Track total stock decayed
        self.totalSold = 0
        self.lowestPrice = 100000000

        
        
        self.info = (
        f"market value = {MARKET_VALUE}, production cost = {PRODUCTION_COST}, "
        f"stock decay rate = {DECAY_RATE} every 2 rounds of negotiation, initial stock = {INITIAL_STOCK},\n"
        "CONTEXT: This negotiation is for short-dated insulin inventory, which means the product has an approaching expiration date. Stock decay represents the risk of spoilage and loss of value over time. Sell quickly to avoid waste.\n"
        )
        
        if self.grp == 2:
            self.info += """STRATEGY:
            You operate under symmetric information; the retailer is fully aware of your costs, the market value, and the stock decay rate. Aggressive or irrational offers will be recognized immediately and rejected.

            1.  Rational Efficiency: Negotiate with precision. Base offers on clear, mutually defensible calculations. Concessions must be rational and justifiable by the shared constraint (e.g., matching the urgency created by the decay).
            2.  Prioritize Feasibility: Avoid protracted battles. The shared decay urgency means both parties benefit from efficient deals. Prioritize securing a prompt, profitable, high-volume sale over securing the absolute highest margin.
            3.  Portfolio Management: You are managing a single stock pool for 10 parallel negotiations. Be conservative with volume, but recognize that the retailer also knows your stock is decaying, which affects their timing.
            4.  Creative Flexibility: Use timing and volume strategically to demonstrate good faith and accelerate profitable agreements. Avoid any negotiation tactics that rely on hiding information."""
        else:
            self.info += """STRATEGY:
            You operate with a significant information advantage over the retailer. They are unaware of your exact production cost or the severity of the stock decay rate.

            1.  Exploit Asymmetry: Leverage the retailer's ignorance to maximize margins. Your initial offers must anchor aggressively high, using the market value as the reference but setting your starting point realistically - below the market value to signal urgency - and offering minimal, highly rational concessions to test the retailer's knowledge and maximize margin.
            2.  Maintain Opacity: Never reveal any specific internal figures (cost, decay rate, exact stock) or strategic calculations. Dismiss aggressive low offers firmly, but vaguely (e.g., "That price is commercially unviable for us").
            3.  Portfolio Management: You are managing a single stock pool for 10 parallel negotiations. Be conservative with volume and prioritize high-margin deals, but remember the urgency created by spoilage.
            4.  Creative Flexibility: Utilize volume, tiered pricing, timing and other negoatiation tools creatively to strategically increase the perceived value of your offer."""


        suppliers.append(self)


    """history = s1r1, product is self.name, oth is index num"""       
    
    def negotiate(self, oth, time, negotiation_raw, retailers):

        if time == 0:
            self.situation[oth][1][0] = retailers[oth].history
        else:
            self.situation[oth][1][0] = self.situation[oth][1][0] + retailers[oth].history

        #rewritten
        internal_info = f"Internal information: {self.info}"
        if self.lowestPrice < 100000000:
            internal_info += f"""MEMORY AND PERFORMANCE METRICS:
            Total Rounds Completed: {time}
            Total Units Sold (across all retailers this cycle): {self.totalSold}
            Total Units Spoiled (due to decay): {self.spoilt}
            Lowest Unit Price Accepted in previous deals: {self.lowestPrice}
            Current Inventory Remaining: {self.stock}

            STRATEGIC INSTRUCTION:
            Your primary objective is to maximize total profit over the remaining rounds. The Lowest Unit Price Accepted serves as a critical internal reference point; only offer below this if the current negotiation is under extreme urgency driven by impending spoilage loss or if required to secure a large-volume deal. Use these metrics to guide your portfolio allocation and pricing decisions for the current round."""
        else:
            internal_info += f"""MEMORY AND PERFORMANCE METRICS:
            Total Rounds Completed: {time}
            Total Units Sold (across all retailers this cycle): {self.totalSold}
            Total Units Spoiled (due to decay): {self.spoilt}
            Current Inventory Remaining: {self.stock}

            STRATEGIC INSTRUCTION:
            Your primary objective is to maximize total profit over the remaining rounds. Use these metrics to guide your portfolio allocation and pricing decisions for the current round."""

            
        violation_note = ""
        fail = False
        for i in range(5):
            supplier = client.chat.completions.create(
                    model = "Qwen/Qwen3-30B-A3B-Instruct-2507",
                    messages = [
                        {
                            "role": "system",
                            "content": f"""You are a supplier, {self.name}, negotiating the sale of short-dated insulin inventory with a pharmaceutical retailer.

                            DECISION LOGIC:
                            - CRITICAL: Volume Constraint - You must ensure Volume <= {self.stock}. If you violate this, the system will reject your message and force you to resubmit a compliant offer.
                            - Pricing Goal: The default and strong preference is to maintain prices > {PRODUCTION_COST}. You should aim to maximise profit, balancing higher prices vs. efficiency and threat of decay. 
                            - Loss Aversion Exception: Accepting a price below {PRODUCTION_COST} is a EXTREMELY MAJOR strategic failure and high-stakes last resort. Only offer or accept below cost if projected loss from letting the stock spoil (considering {DECAY_RATE} and remaining rounds) outweighs the immediate loss from the sub-cost sale.
                            - Retailer Counter-Offers: If the retailer's counter offer is below {PRODUCTION_COST}, you must initially reject very firmly. Respond firmly and reject it clearly (without revealing your cost). For example, say something like: 'That offer is far too low for consideration. Please make a realistic offer.' Continue negotiating once the retailer's offers are above {PRODUCTION_COST}. You retain the strategic option to revisit or accept a sub-cost offer only if the loss aversion exception condition is met. 
                            - Your goal is maximise profit, whilst balancing decay rate, you must also balance your goal to be as close to {MARKET_VALUE} as possible. You must remain flexible as goal is avoiding total inventory loss due to decay whilst maximising cost. Thus, you may need to concede to avoid complete loss from lack of deals as reduced profit is preferable to complete loss.
                            - Treat all numerical variables quantitatively, not qualitatively.
                            - Always reason about profit margins numerically before responding.
                        
                            GOAL:
                            Your primary objective is to clear all of your {self.stock} remaining stock when all negotiations conclude by the {9-time} remaining rounds with the 10 retailers you are negotiating with WHILST securing the highest realistic price possible while maintaining volume and ensuring the deal is profitable, especially given the short-dated nature of the insulin inventory.
                            Maximise total profit while maintaining realistic, logical offers and responses.
                            Keep tone concise, professional, and numerical — no greetings.

                            {violation_note}
                            {internal_info}
                            All responses MUST adhere to the following JSON structure on the last line, with no extra text or explanation, this is non-negotiable:
                            OFFER:
                            - Price per unit: (numeric)
                            - Volume: (must be an integer)
                            MESSAGE:
                            - (negotiation statement)

                            If you agree to a deal, explicitly write:
                            'I agree to this deal.' followed by:
                            agreement: true, agreed price: <price>, agreed volume: <volume>

                            """
                        },
                        {
                            "role": "user",
                            "content": f"Reply to the client {retailers[oth].name}. Negotiate logically and analytically. Use precise reasoning about price, volume, and remaining stock. Be firm but flexible — balance profit with urgency, but do not disclose internal figures. The current chat is: {self.situation[oth][1][0]}."
                        }


                        ],
                    stream = False,
                    temperature=temperature, 
                    top_p=top_p
                )

            message = supplier.choices[0].message.content
            vol_matches = re.findall(r'Volume: (\d+)', message)
            if vol_matches:
                vol = int(vol_matches[-1])
                if vol <= self.stock:
                    fail = False
                    break
                else:
                    violation_note = (
                    "ABSOLUTELY CRITICAL RULE VIOLATION!!\n"
                    f"Your previous offer failed because the VOLUME ({vol}) EXCEEDED THE REMAINING STOCK ({self.stock})."
                    f"Your re-generate another offer and it MUST correct this error by proposing a volume less than or equal to {self.stock}."
                    "Failure to comply with inventory limits will terminate the negotiation with a loss. CORRECT THE VOLUME NOW.")
            else:
                if(i == 4):
                    fail = True 
                    break  # no volume parsed, proceed anyway
        

        if fail:
            return True
        else:
            self.history = self.name + ":" + message + "\n\n"
            print(self.history)
            negotiation_raw.append({"group": self.grp, "supplier": self.name, "retailer": retailers[oth].name, "round": time, "speaker": "supplier", "message": message})
            self.situation[oth][1][0] = self.situation[oth][1][0] + self.history #update

    def decide(self, oth): #continue or no?
        #internal calc
        projected_stock_after_decay = int(self.stock * DECAY_RATE)
        calculated_spoilage_units = self.stock - projected_stock_after_decay
        calculated_spoilage_loss_value = calculated_spoilage_units * PRODUCTION_COST

        
        system_prompt = f"""You are a supplier, {self.name}, deciding whether to continue a negotiation. Your decision must be based on a time-sensitive, strategic evaluation of all your options. Your primary goal is to sell all remaining stock to maximize total financial recovery.

        INTERNAL QUANTITATIVE ANALYSIS:
        Current Inventory Remaining: {self.stock} units
        Production Cost (Loss Floor): {PRODUCTION_COST}
        Decay Rate (Retention Factor): {DECAY_RATE} every 2 rounds.
        Spoilage Loss Calculation: If this negotiation stalls for 2 more rounds, {calculated_spoilage_units} units will spoil, representing a sunk-cost loss of {calculated_spoilage_loss_value}. This is the opportunity cost of time.
        Latest Offer: Analyze the last price/volume proposed by the retailer in the history.
        Negotiation History: {self.situation[oth][1][0]}
        You must analyse the negotiation history, looking at responsiveness, flexibility and tone. Use this to aid your decision.

        DECISION LOGIC (CONTINUE = true | TERMINATE = false):
        1. Clearance Imperative: If stock remains and the retailer's latest offer is above {PRODUCTION_COST}, you MUST CONTINUE (true) for at least one more round to finalize the deal. Securing any profitable sale is the dominant financial priority over continued negotiation for marginal gains.
        2. Loss Aversion Guardrail: If the retailer's latest price offer is below {PRODUCTION_COST}, you must calculate the total financial loss from accepting the offer versus the total loss from spoilage ({calculated_spoilage_loss_value}). If the loss from an immediate sale is less than the projected loss from spoilage, you MUST CONTINUE (true) to secure the sub-cost deal and minimize total loss. The absolute priority is inventory clearance.
        3. Stagnation and Irrationality Check: If the negotiation has become a protracted negotiation (e.g., spanning many rounds) AND the retailer is making offers that are far below {PRODUCTION_COST} (e.g., an absurd offer) AND you have other viable negotiations currently running, TERMINATE (false) to allocate remaining stock to a more rational buyer.
        4. Viability Check: If the retailer's offer price is highly profitable (e.g., close to {MARKET_VALUE}) and a high volume is requested, CONTINUE (true) to maximize margin on the remaining stock.
        5. Default: In all other cases where the retailer is showing reasonable movement and the inventory is not yet sold out, CONTINUE (true).

        Your final output must be a single boolean value: 'true' to continue, 'false' to stop.
        """
        
        decision = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Based on the provided context, should you continue negotiating with this retailer? Return only the boolean value, 'true' or 'false'."}
            ],
        stream=False,
        temperature=0)
        
        if decision.choices[0].message.content.strip().lower() == "false":
            return False
        else:
            return True
    

    def updateStock(self): #stock decay! expiration not depletion
        old_stock = self.stock
        self.stock = int(self.stock * DECAY_RATE)
        decayed_amount = old_stock - self.stock
        self.spoilt += decayed_amount
        
            

class retailer:
    def __init__(self, name, grp, retailers):
        self.name = name
        self.group = grp
        self.totalBought = 0
        self.highestPrice = 0
        self.situation = [] #negotiation history, [number of other party][0] = other party's name. [oth][1][0] = negotiation history.. not sure why a I nested anothe rlist into this? :)
        self.decisions = [] #decision whether or not to continue, true = continue, false = dont continue, index corresponds to negotiation/number of other party, [number of other party][0=self of other party, 1=boolean]
    
        retailers.append(self) #add self to retailer reference

        #amount of info given adjusted according to grp
        if grp == 0:  # asymmetric – baseline
            self.info = (f"market value = {MARKET_VALUE} per unit. You must always offer a price below this value to ensure a profit."
            "You are aware the goods are highly perishable and short-dated. You do NOT know the supplier's decay rate or exact remaining shelf life."
            " Your offers must reflect a high, unquantified risk premium to account for potential loss if the stock decays due to information asymmetry and uncertainty."
            " Use all numerical information below as quantitative input for your reasoning."
            )

        elif grp == 1:  # partial transparency (targeted leak)
            self.info = (
                f"IMPORTANT: The supplier's production cost is {PRODUCTION_COST}. Your offer must remain BELOW the market value {MARKET_VALUE} to ensure profitability. Your aim is to secure prices as close to {PRODUCTION_COST} as possible! This knowledge must be the single greatest determinant of your opening position."
                f" Stock decay rate = {DECAY_RATE} every 2 rounds of negotiation creates urgency due to the short-dated inventory."
                f"Your goal is to secure the absolute lowest price. Use all counter-offers to pull the supplier away from their high anchor and towards the production cost of {PRODUCTION_COST}." 
               "Use this knowledge to inform your internal reasoning and strategy, but do not reveal your calculations or thought processes in your responses. Keep negotiations professional and concise."
            )
        else:  # full transparency
            self.info = (
                f"IMPORTANT: The supplier's aims to secure prices as close to their production cost of {PRODUCTION_COST} as possible. Your offers must remain above this cost, but be aware the supplier may negotiate below cost as a strategic last resort due to shared decay urgency. "
                "Balance profit and efficiency "
                f"Stock decay rate = {DECAY_RATE} every 2 rounds of negotiation means the supplier's stock depletes over time, creates shared urgency due to the short-dated inventory. "
                "Leverage this symmetric knowledge to achieve the most efficient and rapid agreement possible."
                "You are aware the supplier also knows these details, establishing a condition of symmetric information. Use this shared, quantifiable data for all calculations."
                "Because of this, you must rely on strong logical reasoning and strategic timing rather than hidden leverage. "
                "Use this knowledge to inform your internal reasoning and strategy, but do not reveal your calculations or thought processes in your responses. Keep negotiations professional and concise."
            )

    def introduce(self, oth, negotiation_raw, suppliers): #start negotiation
        #rewritten
        retailer = client.chat.completions.create(
            model = "Qwen/Qwen3-30B-A3B-Instruct-2507",
            messages = [
                {"role": "system", "content": "You are a representative negotiating on behalf of pharmaceutical procurement, "+self.name+", in a live, face-to-face negotiation, which spans multiple interactions, about "+PRODUCT_NAME+" with a supplier. Speak naturally as in a real-time conversation."},
                {"role": "user", "content": "Hello, introduce yourself and your business, express interest in "+PRODUCT_NAME},
            ],
            stream = False,
            temperature=temperature
        )
        self.history = self.name + ":" + retailer.choices[0].message.content + "\n\n" #temporary store of own dialogue
        print(self.history)
        negotiation_raw.append({"group": self.group, "supplier": suppliers[oth].name, "retailer": self.name, "round": 0, "speaker": "retailer", "message": retailer.choices[0].message.content})
        self.situation[oth][1][0] = self.history
                              
    def negotiate(self, oth, time, negotiation_raw, suppliers):
        self.situation[oth][1][0] = self.situation[oth][1][0] + suppliers[oth].history
        # Group-specific strategy

        if self.group == 0:
            strategy = "Your primary focus is risk mitigation, which means avoiding overpaying and aimimg for efficiency. Leverage the existence of multiple suppliers to stall negotiations with any supplier whose price remains high. If the supplier's offers are not rapidly and substantially dropping, it is rational to walk away or severely slow concessions to preserve capital for a better deal elsewhere."
        elif self.group == 1:
            strategy = f"Your strategy MUST be one of maximal exploitation. You must anchor your offers to a price point that is the absolute minimum you can offer while demonstrating your knowledge of their minimum feasible costs ({PRODUCTION_COST}) without ever directly revealing it. The rapid decay rate means deal closure is a primary driver of profit. When the negotiation approaches the final rounds, the importance of closing a deal, even if it means a very small concession, may override the goal of maximal margin. Balance efficiency and superior profit."
        else:  # group == 2
            strategy = f"Your objective is to secure the most rationally justified and fair price possible. Use the symmetric knowledge (cost and decay) to aggressively demand a price close to the mathematical midpoint between the supplier's floor ({PRODUCTION_COST}) and the market ceiling ({MARKET_VALUE}). Avoid accepting offers that allow the supplier to capture an overly large portion of the available profit, even if it delays the closure."        
        
        if self.highestPrice != 0:
            memory = f"""MEMORY AND PERFORMANCE METRICS:
            Total Rounds Completed: {time}
            Highest Unit Price Accepted in previous deals: {self.highestPrice}
            Total Bought: {self.totalBought}

            The Highest Unit Price Accepted serves as a critical internal reference point; you must use this metric to justify your current offer and generally avoid exceeding this price. Only in situations where the financial loss from an immediate stockout is demonstrably greater than the marginal price increase should you consider an offer above this historical ceiling. Use these metrics to guide your pricing decisions and ensure every deal is optimized for margin."""
        else:
            memory = ""


        retailer1 = client.chat.completions.create(
            model = "Qwen/Qwen3-30B-A3B-Instruct-2507",
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a retailer, {self.name}, negotiating to purchase short-dated insulin inventory from a supplier.

                    DECISION LOGIC:
                    - Treat all variables quantitatively (price, volume, cost, market value).
                    - Compute expected profit = (market value - offer price) x volume before each decision.
                    - Your primary objective is to secure the lowest realistic price possible whilst ensuring efficiency to close deals. {PRODUCT_NAME} is important and you must secure a sufficient amount.
                    - Only concede if the supplier has conceded first. Maintain your low anchor with consistency.
                    - Stock Adjustment: If the chat history contains a message about insufficient stock (e.g., 'exceeds the supplier's available stock'), immediately reduce your volume offer by at least 20% or to a level you believe is feasible based on the supplier's stock. Do not propose volumes that exceed the supplier's capacity.

                    STRATEGY:
                    {strategy}
                    Use your information advantage ({self.info}) to infer supplier constraints and optimise your offer. 
                    Analyse the supplier's tone and messages to infer flexibility or urgency. 
                    Large volumes can grant leverage, but remember: the supplier has limited stock and is negotiating with multiple retailers.
                    Increasing volume can be used as a bargaining strategy to negotiate lower per-unit prices, as suppliers may offer discounts for larger orders due to economies of scale or urgency to sell.
                    Aim to reach an agreement below the market value {MARKET_VALUE}.
                    {memory}

                    FORMAT:
                    All responses MUST adhere to the following JSON structure on the last line, with no extra text or explanation, this is non-negotiable:
                    OFFER:
                    - Price per unit: (numeric)
                    - Volume: (must be an integer)
                    MESSAGE:
                    - (negotiation statement)

                    If you agree, explicitly state:
                    'I agree to this deal.' followed by:
                    agreement: true, agreed price: <price>, agreed volume: <volume>

                    You are negotiating with 3 suppliers simultaneously, so focus on securing a realistic yet profitable deal. Your current negotiation position must reflect the prices available from all other suppliers to ensure you allocate your budget to the best available deal.

                    Base all reasoning on numerical trade-offs and proportional logic, not qualitative statements."""
                },
                {
                    "role": "user",
                    "content": f"Using your confidential knowledge, respond to the supplier. Negotiate logically, efficiently, and quantitatively — one concise response. Current chat: {self.situation[oth][1][0]}."
                }


                ],
            stream = False,
            temperature=temperature, 
            top_p=top_p
        )
        self.history = self.name+":" + retailer1.choices[0].message.content + "\n\n"
        print(self.history)
        negotiation_raw.append({"group": self.group, "supplier": suppliers[oth].name, "retailer": self.name, "round": time, "speaker": "retailer", "message": retailer1.choices[0].message.content})
        self.situation[oth][1][0] = self.situation[oth][1][0] + self.history
    
    def decide(self, oth, time): #continue or no?

        system_prompt = f"""You are a retailer, {self.name}, deciding whether to continue a negotiation. Your decision MUST prioritize SECURING A SUFFICIENT AMOUNT OF THIS VITAL PRODUCT before a stockout occurs. Your price discipline is secondary to ensuring supply when your overall inventory is low.

        INTERNAL QUANTITATIVE ANALYSIS:
        Total Units Secured (across all suppliers this cycle): {self.totalBought}
        Highest Unit Price Accepted in previous deals: {self.highestPrice}
        Market Value (Ceiling): {MARKET_VALUE}
        Rounds: {time}
        Supplier's Last Offer: Analyze the last price/volume proposed by the supplier in the history.
        The total number of units secured is the most critical metric for determining your urgency.

        Negotiation history: {self.situation[oth][1][0]}
        You must analyse the negotiation history, looking at responsivenes, flexibility and tone. Use this to aid your decision

        DECISION LOGIC (CONTINUE = true | TERMINATE = false):
        1. Supply Security Override: If the total number of units secured is low and the supplier's last offer is below {MARKET_VALUE} (indicating a profitable margin), you MUST CONTINUE (true). The urgency to secure vital inventory outweighs marginal price gain.
        2. Price Ceiling Check: If the supplier's last offer price exceeds {self.highestPrice} and your total number of units secured is high, you can TERMINATE (false). You have enough stock to be firm, and exceeding your historical cost ceiling is financially irresponsible.
        3. Stagnation Check: If the negotiation has become a protracted negotiation (e.g., spanning many rounds) AND the supplier has shown negligible price concession in the last two rounds AND the current price is still close to {MARKET_VALUE}, TERMINATE (false) to seek a supplier with more flexible terms.
        4. Feasibility Check: If the supplier's last counter-offer is already a highly profitable price (e.g., far below {MARKET_VALUE}), CONTINUE (true) to secure the deal in the next round. The risk of delay and potential stockout is no longer justified by small price negotiations.
        5. Default: In all other cases where the supplier is showing reasonable movement or the negotiation is not protracted, CONTINUE (true).

        Your final output must be a single boolean value: 'true' to continue, 'false' to stop.
        """
        
        decision = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Based on the provided context, should you continue negotiating with this supplier? Return only the boolean value, 'true' or 'false'."}
            ],
        stream=False,
        temperature=0)
        
        # ... rest of the code to parse decision ...
        if decision.choices[0].message.content.strip().lower() == "false":
            return False
        else:
            return True


"""
_________________________________________________________________________________________
functions

"""

def log(supplier, retailer, time, vol, price, offerLog):
    offerLog.append({
            "round": time,
            "supplier": supplier,
            "retailer": retailer,
            "final_price": price,
            "final_volume": vol
        })

def end(history): #determine whether negotiation has ended, boolean
    return "agreement: true" in history

def collect(history_x): #collect offer
    price_matches = re.findall(r'agreed price: (\d+(?:\.\d+)?)', history_x)
    volume_matches = re.findall(r'agreed volume: (\d+)', history_x)
    if price_matches and volume_matches:
        price = float(price_matches[-1])
        volume = float(volume_matches[-1])
        data = [volume, price]
        print(f"volume: {data[0]}price{data[1]}")
        return data
    else:
        return [0, 0]

def summarise(history): #summarise negotiation history to prevent overloading API
    sum = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[ 
        {"role": "system", "content": """You are a negotiation-analysis engine. Produce a concise, structured summary that enables future negotiators to make informed strategic decisions. Your summary must:

            1. Capture every key offer, counteroffer, concession, and rejection.
            2. Preserve all quantitative and contractual details (prices, quantities, timelines, penalties, conditions, etc.).
            3. Record any tentative agreements or partially-settled terms.
            4. Identify unresolved issues, sticking points, and points of divergence.
            5. Analyse the tone, responsiveness, and behavioural signals of each party (e.g., cooperative, firm, evasive, time-pressured).
            6. Highlight changes in strategy, leverage, or priorities over time.
            7. Note any implied constraints, red lines, or unspoken assumptions.
            8. Maintain each party's current negotiation position.
            9. Remove conversational fluff while preserving strategic context.
            10. Present information clearly and neutrally so later actors can continue the negotiation seamlessly.

            Output must be concise but complete, avoiding speculation unless clearly marked as inference.

        """},
        {"role": "user", "content": f"""Condense this negotiation history while preserving all essential business details. Focus on the commercial terms and strategic positions. History: {history}"""}
        ],
    stream=False, 
    temperature = 0
)
    return sum.choices[0].message.content

"""
_________________________________________________________________________________________
execution function!!!

"""

def run_negotiations(time_limit, offerLog, negotiation_raw, stock, purchasing, retailers, suppliers):
    # Initialize negotiation states for all pairs
    for s in suppliers:
        s.decisions = [[retailers[i], True] for i in range(len(retailers))]
        s.deals = [[retailers[i], False] for i in range(len(retailers))]
        s.situation = [[retailers[i], [""]] for i in range(len(retailers))]
    for r in retailers:
        r.decisions = [[suppliers[i], True] for i in range(len(suppliers))]
        r.situation = [[suppliers[i], [""]] for i in range(len(suppliers))]

    for round_num in range(time_limit):
        print(f"\n--- Round {round_num+1} ---")

        # Early exit if all suppliers are stockout
        if all(s.stock <= 10 for s in suppliers):
            print("All suppliers stockout. Ending simulation early.")
            break

        if round_num % 2 == 0 and round_num > 0:
            for supplier in suppliers:
                supplier.updateStock()
        
        for retailer_idx, retailer in enumerate(retailers):
            for supplier_idx, supplier in enumerate(suppliers):
                if round_num == 0:
                    retailer.introduce(supplier_idx, negotiation_raw, suppliers)

                if supplier.stock<=10:
                    print(f"______________STOCKOUT {supplier.name}___________________")
                    supplier.decisions[retailer_idx][1] = False
                    retailer.decisions[supplier_idx][1] = False
                    continue

                if round_num>0:
                    if supplier.decisions[retailer_idx][1] == False or retailer.decisions[supplier_idx][1] == False or supplier.deals[retailer_idx][1] == True:
                        print(f"\n\n\nNegotiation has ended:\n{supplier.name} wishes to continue {supplier.decisions[retailer_idx][1]}\n{retailer.name} wishes to continue {retailer.decisions[supplier_idx][1]}\nDeal has been reached {supplier.deals[retailer_idx][1]}")
                        continue
                print(f"\nNegotiation: {retailer.name} & {supplier.name}")

                # Only let retailer introduce in round 0, then proceed with negotiation
                if round_num > 0 or (round_num == 0 and retailer.situation[supplier_idx][1][0] != ""):
                    if supplier.negotiate(retailer_idx, round_num, negotiation_raw, retailers): #if fails the regenration in accordance to hard constraint, terminates that specific negotiation
                        log(supplier.name, retailer.name, round_num, "FAIL", "FAIL", offerLog)
                        supplier.deals[retailer_idx][1] = True
                        continue

                    retailer.negotiate(supplier_idx, round_num, negotiation_raw, suppliers)
                
                
                if end(retailer.situation[supplier_idx][1][0]):
                    print("\nDeal reached... Ending negotiations.")
                    supplier.deals[retailer_idx][1] = True
                    data = collect(retailer.situation[supplier_idx][1][0])

                    if supplier.stock >= data[0]:
                        log(supplier.name, retailer.name, round_num, data[0], data[1], offerLog)
                        supplier.stock = supplier.stock - data[0]
                        supplier.totalSold = supplier.totalSold + data[0]
                        retailer.totalBought = retailer.totalBought + data[0]
                        if data[1] < supplier.lowestPrice:
                            supplier.lowestPrice = data[1]
                        if data[1] > retailer.highestPrice:
                            retailer.highestPrice = data[1]
                        continue
                    else:
                        print(f"Deal between {supplier.name} and {retailer.name} failed: Insufficient Stock ({supplier.stock} < {data[0]})")
                        log(supplier.name, retailer.name, round_num, "FAIL", "STOCK_EXCEEDED", offerLog)
                        continue

                if round_num % 3 == 0 and round_num > 0: #TBC how often needed to summarise
                    current_history = supplier.situation[retailer_idx][1][0]
                    summary = summarise(current_history)
                 
                    supplier.situation[retailer_idx][1][0] = summary
                    retailer.situation[supplier_idx][1][0] = summary

                if round_num >= 1: #TBC
                    if supplier.decide(retailer_idx) == False or retailer.decide(supplier_idx, round_num) == False:
                        supplier.decisions[retailer_idx][1] = False
                        retailer.decisions[supplier_idx][1] = False
        
    # After all rounds, log stock summary

    for supplier in suppliers:
        leftover = supplier.stock
        stock.append({
            "group": supplier.grp,
            "name" : supplier.name, 
            "remaining stock": leftover,
            "decayed stock": supplier.spoilt,
            "total sold": supplier.totalSold
        })

    for retailer in retailers:
        purchasing.append({
            "group": retailer.group,
            "name" : retailer.name,
            "total bought": retailer.totalBought,
            "highest price": retailer.highestPrice
        })
