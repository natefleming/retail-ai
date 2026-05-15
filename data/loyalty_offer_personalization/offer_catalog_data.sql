USE IDENTIFIER(:database);

-- 100 hand-curated offers across 10 brands and 7 categories.
-- Mix of margin classes and discount kinds. Dates anchored to current_date()
-- so the dataset stays valid as time passes.

INSERT INTO offer_catalog VALUES
-- Footwear / Nike
('O-0001','Nike Pegasus 40 - 20% Off','Save 20% on Nike Pegasus 40 running shoes. Cushioned daily trainer for road runners.','Nike','Footwear','PERCENT',0.20,'B','{"min_tier":"Silver"}',current_date()-7,current_date()+30,'EVERGREEN'),
('O-0002','Nike Air Max - $30 Off','Take $30 off any Nike Air Max sneaker. Iconic lifestyle silhouette.','Nike','Footwear','DOLLAR_OFF',0.18,'B','{}',current_date()-14,current_date()+30,'EVERGREEN'),
('O-0003','Nike Metcon Training - 15% Off','15% off the Nike Metcon training shoe line. Stability + grip for lifting and HIIT.','Nike','Footwear','PERCENT',0.15,'A','{"min_tier":"Silver"}',current_date()-3,current_date()+45,'EVERGREEN'),
('O-0004','Nike Free Run - BOGO 50%','Buy one pair of Nike Free Run, get the second pair 50% off.','Nike','Footwear','BOGO',0.25,'C','{"min_lifetime_spend":500}',current_date()-2,current_date()+21,'SPRING'),
('O-0005','Nike Limited Drop - Early Access','Premium-tier early access to the latest Nike limited drop. No discount.','Nike','Footwear','PERCENT',0.0,'A','{"min_tier":"Premium"}',current_date()-1,current_date()+14,'EVERGREEN'),

-- Footwear / Adidas
('O-0006','Adidas Gazelle - 25% Off','25% off Adidas Gazelle classic suede sneakers. Retro icon.','Adidas','Footwear','PERCENT',0.25,'C','{}',current_date()-5,current_date()+30,'EVERGREEN'),
('O-0007','Adidas Samba - $25 Off','$25 off Adidas Samba indoor soccer-inspired sneakers.','Adidas','Footwear','DOLLAR_OFF',0.20,'B','{}',current_date()-10,current_date()+30,'EVERGREEN'),
('O-0008','Adidas Ultraboost - 15% Off','15% off Adidas Ultraboost running shoes. Energy-return midsole.','Adidas','Footwear','PERCENT',0.15,'B','{"min_tier":"Silver"}',current_date()-7,current_date()+30,'EVERGREEN'),
('O-0009','Adidas Originals Bundle','Bundle any three Adidas Originals shoes for 30% off.','Adidas','Footwear','TIERED',0.30,'C','{"min_lifetime_spend":300}',current_date()-3,current_date()+30,'FALL'),
('O-0010','Adidas Track Pants - 20% Off','20% off Adidas Originals track pants. Classic three-stripe.','Adidas','Apparel-Bottoms','PERCENT',0.20,'B','{}',current_date()-7,current_date()+45,'EVERGREEN'),

-- Activewear / Lululemon
('O-0011','Lululemon Align Leggings - 15% Off','15% off Lululemon Align leggings. Buttery-soft Nulu fabric.','Lululemon','Activewear','PERCENT',0.15,'A','{"min_tier":"Gold"}',current_date()-5,current_date()+21,'EVERGREEN'),
('O-0012','Lululemon ABC Pants - $50 Off','$50 off the Lululemon ABC commuter pants.','Lululemon','Apparel-Bottoms','DOLLAR_OFF',0.22,'B','{"min_tier":"Silver"}',current_date()-7,current_date()+30,'EVERGREEN'),
('O-0013','Lululemon Define Jacket - 20% Off','20% off the Lululemon Define studio jacket.','Lululemon','Outerwear','PERCENT',0.20,'B','{}',current_date()-10,current_date()+30,'FALL'),
('O-0014','Lululemon Free Shipping','Free expedited shipping on any Lululemon purchase $80+.','Lululemon','Activewear','FREE_SHIPPING',0.05,'A','{}',current_date()-3,current_date()+60,'EVERGREEN'),
('O-0015','Lululemon Mens Pace Breaker - 15% Off','15% off Pace Breaker training shorts.','Lululemon','Activewear','PERCENT',0.15,'B','{}',current_date()-2,current_date()+30,'SUMMER'),

-- Outerwear / Patagonia
('O-0016','Patagonia Nano Puff - 20% Off','20% off the Patagonia Nano Puff jacket. Lightweight synthetic insulation.','Patagonia','Outerwear','PERCENT',0.20,'B','{"min_tier":"Silver"}',current_date()-7,current_date()+45,'FALL'),
('O-0017','Patagonia Better Sweater - 25% Off','25% off the Patagonia Better Sweater fleece.','Patagonia','Outerwear','PERCENT',0.25,'C','{}',current_date()-5,current_date()+30,'FALL'),
('O-0018','Patagonia Baggies Shorts - 15% Off','15% off Patagonia Baggies shorts. Versatile 5" shorts for hiking + lounging.','Patagonia','Apparel-Bottoms','PERCENT',0.15,'A','{}',current_date()-3,current_date()+45,'SUMMER'),
('O-0019','Patagonia Worn Wear Trade-In','Trade in a used Patagonia item for $30 store credit.','Patagonia','Apparel-Tops','DOLLAR_OFF',0.10,'A','{}',current_date()-30,current_date()+365,'EVERGREEN'),
('O-0020','Patagonia R1 Pullover - $40 Off','$40 off the Patagonia R1 technical fleece pullover.','Patagonia','Outerwear','DOLLAR_OFF',0.20,'B','{"min_tier":"Silver"}',current_date()-10,current_date()+30,'WINTER'),

-- Outerwear / REI
('O-0021','REI Co-op Half Dome Tent - 15% Off','15% off the REI Co-op Half Dome 2 tent.','REI','Accessories','PERCENT',0.15,'B','{"min_tier":"Silver"}',current_date()-7,current_date()+45,'SPRING'),
('O-0022','REI Co-op Rain Jacket - 20% Off','20% off REI Co-op Rainier rain jacket.','REI','Outerwear','PERCENT',0.20,'B','{}',current_date()-5,current_date()+30,'SPRING'),
('O-0023','REI Hiking Socks - BOGO Free','Buy one pair of REI Co-op merino hiking socks, get one free.','REI','Accessories','BOGO',0.50,'C','{}',current_date()-3,current_date()+21,'EVERGREEN'),
('O-0024','REI Backpack - $50 Off','$50 off any REI Co-op trail backpack 30L+.','REI','Accessories','DOLLAR_OFF',0.20,'B','{}',current_date()-7,current_date()+30,'SUMMER'),
('O-0025','REI Co-op Member Bonus','10% Member bonus on any REI Co-op branded item.','REI','Apparel-Tops','PERCENT',0.10,'A','{"min_tier":"Silver"}',current_date()-14,current_date()+90,'EVERGREEN'),

-- Denim / Levis
('O-0026','Levis 501 - 25% Off','25% off the iconic Levi''s 501 straight-leg jeans.','Levis','Denim','PERCENT',0.25,'C','{}',current_date()-7,current_date()+30,'EVERGREEN'),
('O-0027','Levis 511 Slim - $20 Off','$20 off Levi''s 511 slim-fit jeans.','Levis','Denim','DOLLAR_OFF',0.22,'B','{}',current_date()-5,current_date()+30,'EVERGREEN'),
('O-0028','Levis Trucker Jacket - 20% Off','20% off the Levi''s trucker denim jacket. Heritage workwear.','Levis','Outerwear','PERCENT',0.20,'B','{}',current_date()-10,current_date()+30,'FALL'),
('O-0029','Levis Wedgie - 15% Off','15% off Levi''s Wedgie high-rise jeans.','Levis','Denim','PERCENT',0.15,'B','{}',current_date()-3,current_date()+45,'EVERGREEN'),
('O-0030','Levis Vintage Bundle','Buy any 3 Levi''s Vintage Clothing pieces, save 30%.','Levis','Denim','TIERED',0.30,'C','{"min_lifetime_spend":400}',current_date()-7,current_date()+30,'FALL'),

-- Apparel-Tops / GAP
('O-0031','GAP Crewneck Tees - 3 for $30','Three GAP crewneck tees for $30. Mix and match colors.','GAP','Apparel-Tops','TIERED',0.30,'C','{}',current_date()-5,current_date()+30,'EVERGREEN'),
('O-0032','GAP Hoodies - 30% Off','30% off all GAP logo hoodies.','GAP','Apparel-Tops','PERCENT',0.30,'C','{}',current_date()-7,current_date()+30,'FALL'),
('O-0033','GAP Khakis - $25 Off','$25 off any pair of GAP khakis.','GAP','Apparel-Bottoms','DOLLAR_OFF',0.30,'C','{}',current_date()-3,current_date()+30,'EVERGREEN'),
('O-0034','GAP Free Shipping','Free shipping on any GAP order. No minimum.','GAP','Apparel-Tops','FREE_SHIPPING',0.05,'A','{}',current_date()-14,current_date()+60,'EVERGREEN'),
('O-0035','GAP New Arrivals - 15% Off','15% off GAP new-arrivals collection.','GAP','Apparel-Tops','PERCENT',0.15,'A','{}',current_date()-2,current_date()+21,'EVERGREEN'),

-- Apparel-Tops / J.Crew
('O-0036','J.Crew Oxford Shirts - 20% Off','20% off J.Crew oxford button-downs.','JCrew','Apparel-Tops','PERCENT',0.20,'B','{}',current_date()-5,current_date()+30,'EVERGREEN'),
('O-0037','J.Crew Cashmere - 15% Off','15% off J.Crew cashmere sweaters.','JCrew','Apparel-Tops','PERCENT',0.15,'A','{"min_tier":"Silver"}',current_date()-7,current_date()+30,'FALL'),
('O-0038','J.Crew Chinos - $30 Off','$30 off J.Crew 484 slim chinos.','JCrew','Apparel-Bottoms','DOLLAR_OFF',0.25,'B','{}',current_date()-3,current_date()+30,'EVERGREEN'),
('O-0039','J.Crew Tote Bags - 25% Off','25% off all J.Crew canvas tote bags.','JCrew','Accessories','PERCENT',0.25,'C','{}',current_date()-5,current_date()+30,'SUMMER'),
('O-0040','J.Crew Suiting - 30% Off','30% off J.Crew Ludlow suiting separates.','JCrew','Apparel-Tops','PERCENT',0.30,'C','{"min_lifetime_spend":300}',current_date()-7,current_date()+45,'EVERGREEN'),

-- Apparel / Banana Republic
('O-0041','Banana Republic Slim Dress Shirts - 30% Off','30% off Banana Republic non-iron slim dress shirts.','BananaRepublic','Apparel-Tops','PERCENT',0.30,'C','{}',current_date()-5,current_date()+30,'EVERGREEN'),
('O-0042','Banana Republic Wool Suits - $100 Off','$100 off a Banana Republic wool suit.','BananaRepublic','Apparel-Tops','DOLLAR_OFF',0.20,'B','{"min_tier":"Gold"}',current_date()-7,current_date()+30,'EVERGREEN'),
('O-0043','Banana Republic Cashmere - 25% Off','25% off Banana Republic cashmere crewnecks.','BananaRepublic','Apparel-Tops','PERCENT',0.25,'B','{}',current_date()-3,current_date()+30,'FALL'),
('O-0044','Banana Republic Trousers - 20% Off','20% off Banana Republic tailored trousers.','BananaRepublic','Apparel-Bottoms','PERCENT',0.20,'B','{}',current_date()-5,current_date()+30,'EVERGREEN'),
('O-0045','Banana Republic Accessories - BOGO 50%','BOGO 50% off Banana Republic belts and ties.','BananaRepublic','Accessories','BOGO',0.25,'C','{}',current_date()-7,current_date()+30,'EVERGREEN'),

-- Puma
('O-0046','Puma Suede Classic - 20% Off','20% off Puma Suede Classic sneakers.','Puma','Footwear','PERCENT',0.20,'B','{}',current_date()-7,current_date()+30,'EVERGREEN'),
('O-0047','Puma RS-X - $20 Off','$20 off Puma RS-X chunky sneakers.','Puma','Footwear','DOLLAR_OFF',0.18,'B','{}',current_date()-5,current_date()+30,'EVERGREEN'),
('O-0048','Puma Training Tee - 30% Off','30% off Puma performance training tees.','Puma','Activewear','PERCENT',0.30,'C','{}',current_date()-3,current_date()+30,'SUMMER'),
('O-0049','Puma Soccer Cleats - 15% Off','15% off Puma Future and Ultra soccer cleats.','Puma','Footwear','PERCENT',0.15,'A','{"min_tier":"Silver"}',current_date()-7,current_date()+30,'SPRING'),
('O-0050','Puma Track Jacket - 25% Off','25% off Puma T7 track jacket.','Puma','Outerwear','PERCENT',0.25,'B','{}',current_date()-5,current_date()+30,'FALL'),

-- Cross-brand / Category-wide / Evergreen offers
('O-0051','Loyalty Tier Up - Free Reward','Free $25 reward when you reach the next loyalty tier.','ALL_BRANDS','Apparel-Tops','DOLLAR_OFF',0.10,'A','{}',current_date()-30,current_date()+365,'EVERGREEN'),
('O-0052','Birthday Month - 20% Off Sitewide','Birthday-month gift: 20% off your order.','ALL_BRANDS','Apparel-Tops','PERCENT',0.20,'B','{}',current_date()-30,current_date()+365,'EVERGREEN'),
('O-0053','Refer a Friend - $25 Credit','Refer a friend who signs up and earn $25 store credit.','ALL_BRANDS','Apparel-Tops','DOLLAR_OFF',0.10,'A','{}',current_date()-60,current_date()+365,'EVERGREEN'),
('O-0054','Premium Tier Free Shipping','Free shipping on every order. Premium members only.','ALL_BRANDS','Apparel-Tops','FREE_SHIPPING',0.05,'A','{"min_tier":"Premium"}',current_date()-60,current_date()+365,'EVERGREEN'),
('O-0055','Member-Only Flash Sale 30% Off','Member-only flash sale: 30% off select items this weekend.','ALL_BRANDS','Apparel-Tops','PERCENT',0.30,'C','{"min_tier":"Silver"}',current_date()-1,current_date()+3,'EVERGREEN'),
('O-0056','Activewear Bundle - 25% Off','Buy any 3 activewear pieces (any brand) and save 25%.','ALL_BRANDS','Activewear','TIERED',0.25,'B','{}',current_date()-5,current_date()+30,'SPRING'),
('O-0057','Outerwear Clearance - 40% Off','40% off select prior-season outerwear.','ALL_BRANDS','Outerwear','PERCENT',0.40,'C','{}',current_date()-7,current_date()+30,'WINTER'),
('O-0058','New-Arrival Footwear - 10% Off','10% off any newly launched footwear style.','ALL_BRANDS','Footwear','PERCENT',0.10,'A','{}',current_date()-3,current_date()+30,'EVERGREEN'),
('O-0059','Denim Bar Trade-Up','Trade in any old denim for $20 off a new pair.','ALL_BRANDS','Denim','DOLLAR_OFF',0.15,'A','{}',current_date()-30,current_date()+90,'EVERGREEN'),
('O-0060','Accessory Add-On - 50% Off','50% off any accessory when you buy two full-price items.','ALL_BRANDS','Accessories','PERCENT',0.50,'C','{}',current_date()-5,current_date()+30,'EVERGREEN'),

-- Nike additions
('O-0061','Nike Dri-FIT Tees 2 for $40','Two Nike Dri-FIT training tees for $40.','Nike','Activewear','TIERED',0.20,'B','{}',current_date()-5,current_date()+30,'SUMMER'),
('O-0062','Nike Tech Fleece - 20% Off','20% off the Nike Tech Fleece hoodie.','Nike','Apparel-Tops','PERCENT',0.20,'B','{}',current_date()-7,current_date()+30,'FALL'),
('O-0063','Nike Pro Shorts - 15% Off','15% off Nike Pro compression shorts.','Nike','Activewear','PERCENT',0.15,'A','{}',current_date()-3,current_date()+45,'SUMMER'),

-- Adidas additions
('O-0064','Adidas Tiro Track Jacket - $25 Off','$25 off the Adidas Tiro track jacket.','Adidas','Outerwear','DOLLAR_OFF',0.20,'B','{}',current_date()-5,current_date()+30,'FALL'),
('O-0065','Adidas Predator Cleats - 20% Off','20% off Adidas Predator soccer cleats.','Adidas','Footwear','PERCENT',0.20,'B','{"min_tier":"Silver"}',current_date()-7,current_date()+30,'SPRING'),
('O-0066','Adidas Climalite Polo - 25% Off','25% off Adidas Climalite golf polos.','Adidas','Apparel-Tops','PERCENT',0.25,'B','{}',current_date()-3,current_date()+30,'SUMMER'),

-- Patagonia additions
('O-0067','Patagonia Houdini Jacket - 15% Off','15% off the ultra-light Patagonia Houdini windbreaker.','Patagonia','Outerwear','PERCENT',0.15,'A','{}',current_date()-5,current_date()+30,'SPRING'),
('O-0068','Patagonia Capilene Baselayer - 20% Off','20% off Patagonia Capilene midweight baselayers.','Patagonia','Apparel-Tops','PERCENT',0.20,'B','{}',current_date()-7,current_date()+30,'WINTER'),

-- Lululemon additions
('O-0069','Lululemon Wunder Train Bra - 15% Off','15% off the Lululemon Wunder Train sports bra.','Lululemon','Activewear','PERCENT',0.15,'A','{}',current_date()-3,current_date()+30,'EVERGREEN'),
('O-0070','Lululemon Metal Vent Tech SS - 20% Off','20% off the Lululemon Metal Vent Tech SS tee.','Lululemon','Activewear','PERCENT',0.20,'B','{"min_tier":"Silver"}',current_date()-5,current_date()+30,'SUMMER'),

-- Levis additions
('O-0071','Levis 721 High-Rise Skinny - $20 Off','$20 off Levi''s 721 high-rise skinny jeans.','Levis','Denim','DOLLAR_OFF',0.22,'B','{}',current_date()-3,current_date()+30,'EVERGREEN'),
('O-0072','Levis Sherpa Trucker - 25% Off','25% off Levi''s sherpa-lined trucker jacket.','Levis','Outerwear','PERCENT',0.25,'C','{}',current_date()-7,current_date()+30,'WINTER'),

-- GAP additions
('O-0073','GAP Mom Jeans - 20% Off','20% off GAP mom jeans.','GAP','Denim','PERCENT',0.20,'B','{}',current_date()-3,current_date()+30,'EVERGREEN'),
('O-0074','GAP Fleece - BOGO 50%','BOGO 50% off GAP cozy fleece pullovers.','GAP','Apparel-Tops','BOGO',0.25,'C','{}',current_date()-5,current_date()+21,'FALL'),

-- J.Crew additions
('O-0075','J.Crew Beach Shorts - 30% Off','30% off J.Crew board shorts.','JCrew','Apparel-Bottoms','PERCENT',0.30,'C','{}',current_date()-3,current_date()+30,'SUMMER'),
('O-0076','J.Crew Cashmere Beanie - $15 Off','$15 off J.Crew cashmere beanies.','JCrew','Accessories','DOLLAR_OFF',0.30,'C','{}',current_date()-7,current_date()+30,'WINTER'),

-- Banana Republic additions
('O-0077','Banana Republic Linen Shirts - 25% Off','25% off Banana Republic linen short-sleeves.','BananaRepublic','Apparel-Tops','PERCENT',0.25,'B','{}',current_date()-3,current_date()+30,'SUMMER'),
('O-0078','Banana Republic Loafers - $40 Off','$40 off Banana Republic premium loafers.','BananaRepublic','Footwear','DOLLAR_OFF',0.20,'B','{"min_tier":"Gold"}',current_date()-5,current_date()+30,'EVERGREEN'),

-- Puma additions
('O-0079','Puma Mayze Platform - 20% Off','20% off the Puma Mayze chunky-platform sneaker.','Puma','Footwear','PERCENT',0.20,'B','{}',current_date()-3,current_date()+30,'EVERGREEN'),
('O-0080','Puma x BMW Motorsport Jacket - 30% Off','30% off Puma x BMW Motorsport jacket.','Puma','Outerwear','PERCENT',0.30,'C','{"min_lifetime_spend":300}',current_date()-7,current_date()+30,'EVERGREEN'),

-- REI additions
('O-0081','REI Trail Running Shoes - 15% Off','15% off REI Co-op trail running shoes.','REI','Footwear','PERCENT',0.15,'A','{}',current_date()-3,current_date()+30,'SPRING'),
('O-0082','REI Hydration Pack - 20% Off','20% off REI Co-op hydration packs.','REI','Accessories','PERCENT',0.20,'B','{}',current_date()-5,current_date()+30,'SUMMER'),

-- Cross-brand seasonal + behavior-triggered
('O-0083','Winter Coat Sale - 35% Off','35% off select winter coats from premium brands.','ALL_BRANDS','Outerwear','PERCENT',0.35,'C','{}',current_date()-7,current_date()+30,'WINTER'),
('O-0084','Spring Refresh Bundle','Spring refresh: buy any top + bottom, save 20% on both.','ALL_BRANDS','Apparel-Tops','TIERED',0.20,'B','{}',current_date()-3,current_date()+30,'SPRING'),
('O-0085','Back-to-School - $30 Off $100','$30 off any $100+ order. Back-to-school promo.','ALL_BRANDS','Apparel-Tops','DOLLAR_OFF',0.20,'B','{}',current_date()-5,current_date()+30,'FALL'),
('O-0086','Cyber Week 25% Off','Cyber Week: 25% off most full-price items.','ALL_BRANDS','Apparel-Tops','PERCENT',0.25,'C','{}',current_date()-1,current_date()+5,'WINTER'),
('O-0087','Cardholder Bonus - 5x Points','Earn 5x loyalty points on every purchase this week.','ALL_BRANDS','Apparel-Tops','DOLLAR_OFF',0.05,'A','{"min_tier":"Silver"}',current_date()-2,current_date()+7,'EVERGREEN'),
('O-0088','Lapsed-Member Comeback - 20% Off','We miss you: 20% off your first order back.','ALL_BRANDS','Apparel-Tops','PERCENT',0.20,'B','{}',current_date()-14,current_date()+60,'EVERGREEN'),
('O-0089','VIP Stylist Session - Free','Complimentary 30-minute virtual styling session.','ALL_BRANDS','Apparel-Tops','DOLLAR_OFF',0.0,'A','{"min_tier":"Premium"}',current_date()-30,current_date()+365,'EVERGREEN'),
('O-0090','Eco-Conscious Pick - 10% Off','10% off any item from the responsibly-sourced collection.','ALL_BRANDS','Apparel-Tops','PERCENT',0.10,'A','{}',current_date()-7,current_date()+60,'EVERGREEN'),

-- Niche / brand-specific final ten
('O-0091','Nike SB Skate Bundle - 25% Off','25% off any 3 Nike SB items.','Nike','Footwear','TIERED',0.25,'C','{}',current_date()-3,current_date()+30,'EVERGREEN'),
('O-0092','Adidas Yeezy Notify - Early Access','Early-access notification for the next Yeezy drop. Premium tier.','Adidas','Footwear','PERCENT',0.0,'A','{"min_tier":"Premium"}',current_date()-7,current_date()+30,'EVERGREEN'),
('O-0093','Lululemon Mens Joggers - 15% Off','15% off Lululemon ABC joggers.','Lululemon','Apparel-Bottoms','PERCENT',0.15,'A','{}',current_date()-3,current_date()+30,'EVERGREEN'),
('O-0094','Patagonia Down Sweater - 20% Off','20% off Patagonia Down Sweater jacket.','Patagonia','Outerwear','PERCENT',0.20,'B','{"min_tier":"Silver"}',current_date()-5,current_date()+30,'WINTER'),
('O-0095','REI Co-op Camp Chair - 15% Off','15% off REI Co-op flexlite camp chair.','REI','Accessories','PERCENT',0.15,'B','{}',current_date()-7,current_date()+45,'SUMMER'),
('O-0096','Levis Western Snap Shirt - 25% Off','25% off Levi''s Western snap-front shirts.','Levis','Apparel-Tops','PERCENT',0.25,'B','{}',current_date()-3,current_date()+30,'EVERGREEN'),
('O-0097','GAP Maternity - 20% Off','20% off GAP maternity collection.','GAP','Apparel-Tops','PERCENT',0.20,'B','{}',current_date()-7,current_date()+30,'EVERGREEN'),
('O-0098','J.Crew Wedding Guest - 25% Off','25% off wedding-guest dresses and suiting at J.Crew.','JCrew','Apparel-Tops','PERCENT',0.25,'B','{}',current_date()-3,current_date()+30,'SPRING'),
('O-0099','Banana Republic Travel Capsule','Banana Republic travel capsule: pick any 5 pieces for $250.','BananaRepublic','Apparel-Tops','TIERED',0.30,'C','{"min_lifetime_spend":500}',current_date()-7,current_date()+30,'SUMMER'),
('O-0100','Puma Golf Shoes - 20% Off','20% off Puma performance golf shoes.','Puma','Footwear','PERCENT',0.20,'B','{"min_tier":"Silver"}',current_date()-5,current_date()+30,'SPRING');
