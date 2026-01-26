"""
Integration tests for instruction reranker with Databricks LLM.

Tests structured output approaches with real Databricks models.

To run integration tests:
    pytest tests/dao_ai/test_instruction_reranker_integration.py -v -m integration
"""

import os

import pytest
from databricks_langchain import ChatDatabricks

from dao_ai.config import RankedDocument, RankingResult

# Check if we have Databricks credentials
HAS_DATABRICKS_CREDS = bool(
    os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN")
)

SKIP_MSG = "Requires DATABRICKS_HOST and DATABRICKS_TOKEN environment variables"

# Test prompt - matches the format used in instruction_reranker.py
TEST_PROMPT = """Rerank these search results for the query "milwaukee drills".

Prioritize results that match explicit user constraints:
- Price constraints: "under $100" → prefer lower-priced items
- Brand preferences: "Milwaukee" → boost Milwaukee products
- Category filters: "power tools" → prefer power tool category
- Recency: "recent" or "new" → prefer recently updated items
When constraints conflict, explicit mentions take priority.


Available metadata fields: product_id (string), sku (string), upc (string), brand_name (string), product_name (string), merchandise_class (string), class_cd (string), description (string)

## Documents

[0] Content: Milwaukee 2-9/16 in. X 6 in. L Heat-Treated Steel Self-Feed Drill Bit Hex Shank 1 pc Milwaukee selfeed drill bits deliver speed and endurance for repetitive drilling of large holes. Designed for any trade that demands woodcutting for installing pipe and conduit, selfeed bits feed into work without p...
    Metadata: chunk_id: 747324309506.0, sku: 02732510, upc: 0273251312769, brand_name: MILWAUKEE, product_name: Milwaukee 2-9/16 in. X 6 in. L Heat-Treated Steel Self-Feed Drill Bit Hex Shank 1 pc, merchandise_class: SELF FEED/EXTENSION BITS, class_cd: 223, score: 0.5677276390008058

[1] Content: Milwaukee 7.5 amps 1/2 in. Corded Hammer Drill The power to drill 1/2 in. holes in concrete. The versatility to excel at wood and steel applications. This powerful 7.5-amp, 1/2 in. Hammer Drill features dual trigger speed control in two ranges, with more power for the toughest applications or more s...
    Metadata: chunk_id: 1486058684588.0, sku: 85493858, upc: 8549385123703, brand_name: MILWAUKEE, product_name: Milwaukee 7.5 amps 1/2 in. Corded Hammer Drill, merchandise_class: HEAVY-DUTY POWER TOOLS, class_cd: 286, score: 0.8961038961038961

[2] Content: Milwaukee 9/64 in. X 7.755 in. L High Speed Steel Drill and Countersink Quick-Change Hex Shank 1 pk Milwaukee #6 Countersink features a 9/64 in. Drill Bit. This drill bit is a two-in-one tool that drills a pilot hole and countersinks it in one step. This is ideal for creating clean, professional-loo...
    Metadata: chunk_id: 446676598955.0, sku: 87199192, upc: 8719919549211, brand_name: MILWAUKEE, product_name: Milwaukee 9/64 in. X 7.755 in. L High Speed Steel Drill and Countersink Quick-Change Hex Shank 1 pk, merchandise_class: POWER DRILL BITS, class_cd: 252, score: 0.8108274647887324

[3] Content: Milwaukee #8 X 6 in. L Step Drill Bit 3-Flat Shank 1 pc Milwaukee Step drill bits with jam-free performance feature a dual-flute design delivering up to 2X faster hole times, up to 4X longer life and up to 50% more holes per battery charge than the competition. Ideal for drilling large and small-dia...
    Metadata: chunk_id: 549755813909.0, sku: 09821583, upc: 0982158302833, brand_name: MILWAUKEE, product_name: Milwaukee #8 X 6 in. L Step Drill Bit 3-Flat Shank 1 pc, merchandise_class: POWER DRILL BITS, class_cd: 252, score: 0.8417396269508945

[4] Content: Milwaukee 2 in. X 6 in. L Heat-Treated Steel Self-Feed Drill Bit Hex Shank 1 pc Milwaukee selfeed drill bits deliver speed and endurance for repetitive drilling of large holes. Designed for any trade that demands woodcutting for installing pipe and conduit, selfeed bits feed into work without pressu...
    Metadata: chunk_id: 249108103317.0, sku: 83598180, upc: 8359818937650, brand_name: MILWAUKEE, product_name: Milwaukee 2 in. X 6 in. L Heat-Treated Steel Self-Feed Drill Bit Hex Shank 1 pc, merchandise_class: SELF FEED/EXTENSION BITS, class_cd: 223, score: 0.5677276390008058

[5] Content: Milwaukee #9 X 6 in. L Step Drill Bit 3-Flat Shank 1 pc Milwaukee Step drill bits with jam-free performance feature a dual-flute design delivering up to 2X faster hole times, up to 4X longer life and up to 50% more holes per battery charge than the competition. Ideal for drilling large and small-dia...
    Metadata: chunk_id: 1520418422871.0, sku: 43942466, upc: 4394246374961, brand_name: MILWAUKEE, product_name: Milwaukee #9 X 6 in. L Step Drill Bit 3-Flat Shank 1 pc, merchandise_class: POWER DRILL BITS, class_cd: 252, score: 0.8531885758998434

[6] Content: Milwaukee M12 3/8 in. Brushed Cordless Drill/Driver Kit (Battery & Charger) The Milwaukee M12 3/8 in. Drill/Driver Kit drills and fastens up to 35% faster than competitors and is the only tool in its class with an all-metal locking chuck. The powerful compact cordless drill driver delivers 275 in. l...
    Metadata: chunk_id: 249108103288.0, sku: 64349640, upc: 6434964953142, brand_name: MILWAUKEE, product_name: Milwaukee M12 3/8 in. Brushed Cordless Drill/Driver Kit (Battery & Charger), merchandise_class: HEAVY-DUTY POWER TOOLS, class_cd: 286, score: 0.6549400342661336

[7] Content: Milwaukee 7/8 - 1-3/8 in. Black Oxide Step Drill Bit 1 pk Milwaukee step drill bit with Jam-Free performance features a dual-flute design delivering up to 2X faster hole times, up to 4X longer life, and up to 50% more holes per battery charge than competitors. Ideal for drilling small and large-diam...
    Metadata: chunk_id: 1589137899691.0, sku: 83642500, upc: 8364250609485, brand_name: MILWAUKEE, product_name: Milwaukee 7/8 - 1-3/8 in. Black Oxide Step Drill Bit 1 pk, merchandise_class: POWER DRILL BITS, class_cd: 252, score: 0.9215129486597001

[8] Content: Milwaukee M18 1/2 in. Brushless Cordless Hammer Drill Tool Only The Milwaukee M18 1/2 in. Brushless Hammer Drill/Driver delivers the most power in its class while maintaining compact size delivering longer run-time. The brushless motor was built and optimized specifically for the tool. Paired with b...
    Metadata: chunk_id: 188978561188.0, sku: 81078844, upc: 8107884696168, brand_name: MILWAUKEE, product_name: Milwaukee M18 1/2 in. Brushless Cordless Hammer Drill Tool Only, merchandise_class: HEAVY-DUTY POWER TOOLS, class_cd: 286, score: 0.6224971008890607

[9] Content: Milwaukee 8 amps 1/2 in. Corded Hammer Drill The most powerful single-speed hammer drill on the market, with up to 2X more durability. This practical 8-amp, 1/2 in. can be used in hammer mode and in regular drill mode. The hammer drill offers power and versatility for a variety of tough jobs, with a...
    Metadata: chunk_id: 1005022347342.0, sku: 45689079, upc: 4568907352647, brand_name: MILWAUKEE, product_name: Milwaukee 8 amps 1/2 in. Corded Hammer Drill, merchandise_class: HEAVY-DUTY POWER TOOLS, class_cd: 286, score: 0.7766317485898468

[10] Content: Milwaukee Jam-Free 3/16 - 7/8 in. X 6 in. L Metal Step Drill Bit 3-Flat Shank 1 pc Milwaukee Step drill bits with jam-free performance feature a dual-flute design delivering up to 2X faster hole times, up to 4X longer life and up to 50% more holes per battery charge than the competition. Ideal for d...
    Metadata: chunk_id: 429496729741.0, sku: 77773333, upc: 7777333924664, brand_name: MILWAUKEE, product_name: Milwaukee Jam-Free 3/16 - 7/8 in. X 6 in. L Metal Step Drill Bit 3-Flat Shank 1 pc, merchandise_class: POWER DRILL BITS, class_cd: 252, score: 0.836244131455399

[11] Content: Milwaukee 12 in. Alloy Steel Drill Bit Extension 7/16 in. Hex Shank 1 pc Extend your reach by 12 in. when drilling with Milwaukee selfeed bits, auger bits and hole saws. Useful for drilling into deep holes, tight spaces, or uneven surfaces, secures with a hex key included.. The 7/16 inch hex fits bo...
    Metadata: chunk_id: 1022202216467.0, sku: 10629147, upc: 1062914399242, brand_name: MILWAUKEE, product_name: Milwaukee 12 in. Alloy Steel Drill Bit Extension 7/16 in. Hex Shank 1 pc, merchandise_class: SELF FEED/EXTENSION BITS, class_cd: 223, score: 0.6939883645765998

[12] Content: Milwaukee 0.3 in. L High Speed Steel Drill and Countersink Set Quick-Change Hex Shank 3 pc The Milwaukee 3 pc countersink bit set features #6, #8, and #10 countersinks. This drill bit is a two-in-one tool that drills a pilot hole and countersinks it in one step. This is ideal for creating clean, pro...
    Metadata: chunk_id: 1503238553635.0, sku: 19417493, upc: 1941749816165, brand_name: MILWAUKEE, product_name: Milwaukee 0.3 in. L High Speed Steel Drill and Countersink Set Quick-Change Hex Shank 3 pc, merchandise_class: POWER DRILL BITS, class_cd: 252, score: 0.8860759493670887

[13] Content: Milwaukee 10 X 3/16 in. D Black Oxide Countersink Bit 1 pc The Milwaukee 10 countersink features a 3/16 in drill bit. The 1/4 in. Quick Change hex shanks allow for quick size changes. The 1/8 in. hex drill bit length is adjustable to allow for precise depth control. Category: POWER DRILL BITS; Brand...
    Metadata: chunk_id: 1271310319659.0, sku: 27533215, upc: 2753321632440, brand_name: MILWAUKEE, product_name: Milwaukee 10 X 3/16 in. D Black Oxide Countersink Bit 1 pc, merchandise_class: POWER DRILL BITS, class_cd: 252, score: 0.5782479898434194

[14] Content: Milwaukee 1/2 in. X 3.75 in. L Carbide Tipped Glass/Tile Drill Bit 3-Flat Shank 1 pc Our Glass and Tile Drill Bit features an exact start tip for clean holes and minimal bit walking in glass and ceramic tile. Engineered with sharpened carbide, Milwaukee glass and tile bits provide faster tile drilli...
    Metadata: chunk_id: 1168231104585.0, sku: 42164760, upc: 4216476856447, brand_name: MILWAUKEE, product_name: Milwaukee 1/2 in. X 3.75 in. L Carbide Tipped Glass/Tile Drill Bit 3-Flat Shank 1 pc, merchandise_class: POWER DRILL BITS, class_cd: 252, score: 0.8588235294117648

[15] Content: Milwaukee Shockwave 7/16 in. X 4.92 in. L Titanium Red Helix Drill Bit Hex Shank 1 pc Milwaukee Shockwave impact duty titanium drill bits with Red Helix are engineered for impacts and drills. Designed with a variable Helix that with an aggressive 35 degree cutting edge which ends at 15 degree, The q...
    Metadata: chunk_id: 1211180777601.0, sku: 69017513, upc: 6901751470316, brand_name: MILWAUKEE, product_name: Milwaukee Shockwave 7/16 in. X 4.92 in. L Titanium Red Helix Drill Bit Hex Shank 1 pc, merchandise_class: POWER DRILL BITS, class_cd: 252, score: 0.5446428571428572

[16] Content: Milwaukee 3/8 in. X 5 in. L High Speed Steel Brad Point Bits Drill Bit Round Shank 1 pc For drilling the most accurate and clean holes in wood and plywood, choose the Milwaukee 3/8 in. Brad Point Bit. Milwaukee Brad Point Bits start holes with precision and finish without splinters. The machined tip...
    Metadata: chunk_id: 283467841677.0, sku: 87133189, upc: 8713318710074, brand_name: MILWAUKEE, product_name: Milwaukee 3/8 in. X 5 in. L High Speed Steel Brad Point Bits Drill Bit Round Shank 1 pc, merchandise_class: SELF FEED/EXTENSION BITS, class_cd: 223, score: 0.7540455498951153

[17] Content: Milwaukee 1/8 in. X 2-3/4 in. L High Speed Steel Brad Point Bits Drill Bit Round Shank 1 pc For drilling the most accurate and clean holes in wood and plywood, choose the Milwaukee 1/8 in. Brad Point Bit. Milwaukee Brad Point Bits start holes with precision and finish without splinters. The machined...
    Metadata: chunk_id: 1202590843012.0, sku: 70114843, upc: 7011484327043, brand_name: MILWAUKEE, product_name: Milwaukee 1/8 in. X 2-3/4 in. L High Speed Steel Brad Point Bits Drill Bit Round Shank 1 pc, merchandise_class: SELF FEED/EXTENSION BITS, class_cd: 223, score: 0.7472857981220657

[18] Content: Milwaukee 5.25 in. L Carbide Tipped Glass and Tile Bit Set 3-Flat Shank 4 pk The Milwaukee 4-pc glass and tile drill bit set features an exact start tip for clean holes and minimal bit walking in glass and ceramic tile. Engineered with sharpened carbide, Milwaukee glass and tile bits provide faster ...
    Metadata: chunk_id: 249108103296.0, sku: 68054663, upc: 6805466060179, brand_name: MILWAUKEE, product_name: Milwaukee 5.25 in. L Carbide Tipped Glass and Tile Bit Set 3-Flat Shank 4 pk, merchandise_class: POWER DRILL BIT SETS, class_cd: 251, score: 0.8588235294117648

[19] Content: Milwaukee 1/4 in. X 2.25 in. L Carbide Tipped Glass/Tile Drill Bit Round Shank 1 pc Our Glass and Tile Drill Bit features an exact start tip for clean holes and minimal bit walking in glass and ceramic tile. Engineered with sharpened carbide, Milwaukee glass and tile bits provide faster tile drillin...
    Metadata: chunk_id: 927712935981.0, sku: 25341331, upc: 2534133797264, brand_name: MILWAUKEE, product_name: Milwaukee 1/4 in. X 2.25 in. L Carbide Tipped Glass/Tile Drill Bit Round Shank 1 pc, merchandise_class: POWER DRILL BITS, class_cd: 252, score: 0.8765432098765432

Score each document 0.0-1.0 based on relevance to the query and instructions. Only include documents scoring > 0.1, sorted highest first."""


@pytest.mark.integration
@pytest.mark.skipif(not HAS_DATABRICKS_CREDS, reason=SKIP_MSG)
class TestStructuredOutputWithDatabricks:
    """Integration tests for structured output with Databricks LLM."""

    @pytest.fixture
    def llm(self) -> ChatDatabricks:
        """Create ChatDatabricks instance."""
        return ChatDatabricks(
            endpoint="databricks-claude-sonnet-4",
            temperature=0.0,
        )

    def test_with_structured_output_default_method(self, llm: ChatDatabricks) -> None:
        """Test with_structured_output using default method (tool calling)."""
        structured_llm = llm.with_structured_output(RankingResult)

        result = structured_llm.invoke(TEST_PROMPT)

        # Validate result structure
        assert isinstance(result, RankingResult)
        assert isinstance(result.rankings, list)

        # Should have some rankings
        assert len(result.rankings) > 0

        # Validate each ranking
        for ranking in result.rankings:
            assert isinstance(ranking, RankedDocument)
            assert 0 <= ranking.index <= 19  # 20 documents in test
            assert 0.0 <= ranking.score <= 1.0
            assert isinstance(ranking.reason, str)
            assert len(ranking.reason) > 0

        # Note: LLM may not perfectly sort rankings, that's a quality issue not a structured output issue

        print("\n=== Tool Calling Method Results ===")
        print(f"Number of rankings: {len(result.rankings)}")
        for r in result.rankings:
            print(f"  [{r.index}] score={r.score:.2f} - {r.reason[:80]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration", "-s"])
