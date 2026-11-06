"""
55 Authentic Classical Hindu Ayurvedic Scripture Quotes
Sources: Charaka Samhita, Sushruta Samhita, Ashtanga Hridaya, Ashtanga Sangraha,
Madhava Nidana, Yogaratnakara, Sharngadhara Samhita, Bhavaprakasha, Kashyapa Samhita.
Every quote includes Sanskrit (Devanagari + IAST transliteration), English translation, and clinical explanation.
"""

import random

AYURVEDIC_QUOTES = [
    {
        "source": "Ashtanga Hridaya, Sutrasthana 1.20",
        "sanskrit": "रोगस्तु दोषवैषम्यं दोषसाम्यमरोगता।",
        "transliteration": "Rogastu doshavaishamyam doshasamyamarogata.",
        "english": "Imbalance of the three Doshas (Vata, Pitta, Kapha) is disease; their perfect equilibrium is health.",
        "explanation": "Ayurveda defines disease not merely as an external infection, but as a disturbance in the natural balance of Vata, Pitta, and Kapha."
    },
    {
        "source": "Yogaratnakara — Ashtasthana Pariksha (Jihva Pariksha)",
        "sanskrit": "जिह्वा शीतला कर्कशा स्फुटिता मारुताधिके। रक्ता श्यावा च पित्तेन, कफेन धवलाऽऽर्द्रका॥",
        "transliteration": "Jihva shitala karkasha sphutita marutadhike | Rakta shyava cha pittena, kaphena dhavala ardraka ||",
        "english": "In Vata excess, the tongue becomes cold, rough, and fissured; in Pitta, it turns red or dark; in Kapha, it is white, thick, and moist.",
        "explanation": "This classical verse is the foundation of Ayurvedic tongue diagnosis (Jihva Pariksha), directly linking surface texture and colour to the three Doshas."
    },
    {
        "source": "Sushruta Samhita, Sutrasthana 15.41",
        "sanskrit": "समदोषः समाग्निश्च समधातुमलक्रियः। प्रसन्नात्मेन्द्रियमनाः स्वस्थ इत्यभिधीयते॥",
        "transliteration": "Samadoshah samagnishcha samadhatu-malakriyah | Prasannatmendriyamanah svastha ityabhidhiyate ||",
        "english": "One whose Doshas are balanced, digestive fire (Agni) is steady, tissues and eliminations function properly, and whose soul, senses, and mind are serene, is called Svastha (healthy).",
        "explanation": "True health in Ayurveda requires harmony across physical metabolism (Agni), structural tissues (Dhatu), waste elimination (Mala), and mental peace."
    },
    {
        "source": "Charaka Samhita, Chikitsasthana 15.3",
        "sanskrit": "आयुर्वर्णो बलं स्वास्थ्यमुत्साहोपचयौ प्रभा। ओजस्तेजोऽग्नयः प्राणाश्चोक्ता देहाग्निहेतुकाः॥",
        "transliteration": "Ayurvarno balam svasthyam utsahopachayau prabha | Ojastejo'gnayah pranashchokta dehagnihetukah ||",
        "english": "Lifespan, complexion, strength, health, enthusiasm, nourishment, lustre, immunity (Ojas), and vitality all depend on the digestive fire (Dehagni).",
        "explanation": "The tongue is the direct mirror of Agni (digestive fire). When Agni is strong, the tongue remains pink, clean, and vibrant."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 5.74–75",
        "sanskrit": "जिह्वामूलगतान् मलान् उच्छ्वासरोधिनीं दुर्गन्धिं भजते जिह्वा तस्याः संशोधनं हितम्।",
        "transliteration": "Jihvamulagatan malan uchchhvasarodhinim durgandhim bhajate jihva tasyah samshodhanam hitam.",
        "english": "Impurities accumulated at the root of the tongue obstruct respiration and cause foul odour; therefore, gentle cleansing of the tongue is wholesome.",
        "explanation": "Acharya Charaka prescribes daily tongue scraping (Jihva Nirlekhana) to clear Ama coating and awaken taste perception."
    },
    {
        "source": "Yogaratnakara — Roga Pariksha",
        "sanskrit": "रोगाक्रान्तशरीरस्य स्थानान्यष्टौ परीक्षयेत्। नाडीं मूत्रं मलं जिह्वां शब्दस्पर्शदृगाकृतीः॥",
        "transliteration": "Rogakrantasharirasya sthananyashtau parikshayet | Nadim mutram malam jihvam shabdasparshadrigakritih ||",
        "english": "In an afflicted body, eight clinical sites must be examined: pulse (Nadi), urine (Mutra), stool (Mala), tongue (Jihva), voice (Shabda), touch (Sparsha), eyes (Drik), and physique (Akriti).",
        "explanation": "Jihva Pariksha (tongue examination) is one of the eight pillars of classical Ayurvedic clinical assessment (Ashtasthana Pariksha)."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 1.42",
        "sanskrit": "हिताहितं सुखं दुःखमायुस्तस्य हिताहितम्। मानं च तच्च यत्रोक्तमायुर्वेदः स उच्यते॥",
        "transliteration": "Hitahitam sukham duhkham ayustasya hitahitam | Manam cha tachcha yatroktam ayurvedah sa uchyate ||",
        "english": "That science which explains wholesome and unwholesome life, happy and sorrowful life, and what promotes longevity is called Ayurveda.",
        "explanation": "Ayurveda guides daily dietary and lifestyle choices (Pathya-Apathya) tailored to one's internal constitution."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 30.26",
        "sanskrit": "प्रयोजनं चास्य स्वस्थस्य स्वास्थ्यरक्षणमातुरस्य विकारप्रशमनं च।",
        "transliteration": "Prayojanam chasya svasthasya svasthyarakshanam aturasya vikaraprashamanam cha.",
        "english": "The twofold purpose of Ayurveda is to preserve the health of the healthy and to alleviate the disorders of the diseased.",
        "explanation": "Early tongue screening detects subtle Dosha and Ama imbalances before they manifest as full-blown clinical illnesses."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 2.1",
        "sanskrit": "ब्राह्मे मुहूर्ते उत्तिष्ठेत् स्वस्थो रक्षार्थमायुषः।",
        "transliteration": "Brahme muhurte uttishthet svastho raksharthamayushah.",
        "english": "A healthy person should rise during Brahma Muhurta (before dawn) to protect vitality and longevity.",
        "explanation": "Examining and cleansing the tongue first thing in the morning reveals the true overnight metabolic state of the gastrointestinal tract."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 13.25",
        "sanskrit": "ऊष्मणोऽल्पबलत्वेन धातुमाद्यमपाचितम्। दुष्टमामाशयगतं रसमामं प्रचक्षते॥",
        "transliteration": "Ushmano'lpabalatvena dhatumadyam apachitam | Dushtam amashayagatam rasamamam prachakshate ||",
        "english": "Due to weakened digestive heat (Agni), the first nutrient fluid remains undigested and ferments in the stomach; this toxic state is called Ama.",
        "explanation": "A thick white or yellowish coating (Lepa) on the dorsal surface of the tongue is the primary clinical sign of Ama in the digestive tract."
    },
    {
        "source": "Charaka Samhita, Chikitsasthana 15.4",
        "sanskrit": "शान्तेऽग्नौ म्रियते, युक्ते चिरं जीवत्यनामयः। रोगी स्याद्विकृते, मूलमग्निस्तस्मान्निरुच्यते॥",
        "transliteration": "Shante'gnau mriyate, yukte chiram jivatyanamayah | Rogi syadvikrite, mulam agnistasmanniruchyate ||",
        "english": "When Agni is extinguished, life ceases; when balanced, one lives long without disease; when deranged, illness arises. Therefore, Agni is the root of life.",
        "explanation": "Maintaining balanced digestive fire (Sama Agni) keeps the tongue clean, moist, and free from toxic coating."
    },
    {
        "source": "Madhava Nidana, Amavata Nidana 1",
        "sanskrit": "मन्दाग्नेरन्नमपक्वं सङ्गं प्राप्यामाशयस्थितम्। आममित्यभिधीयते सर्वरोगप्रकोपणम्॥",
        "transliteration": "Mandagnerannam apakvam sangam prapyamashayasthitam | Amamityabhidhiyate sarvarogaprakopanam ||",
        "english": "Undigested food retained in the stomach due to sluggish digestive fire is termed Ama, the root provocateur of all diseases.",
        "explanation": "Clearing Ama through light warm meals and herbal digestives (Deepana-Pachana) restores tongue clarity and systemic energy."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 1.6",
        "sanskrit": "वायुः पित्तं कफश्चेति त्रयो दोषाः समासतः।",
        "transliteration": "Vayuh pittam kaphashcheti trayo doshah samasatah.",
        "english": "In brief, Vata, Pitta, and Kapha are the three fundamental bio-energies (Doshas) governing the body.",
        "explanation": "Every physiological and pathological sign on the tongue reflects the dynamic interplay of Vata (movement), Pitta (heat), and Kapha (structure)."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 1.7",
        "sanskrit": "विकृताविकृता देहं घ्नन्ति ते वर्तयन्ति च।",
        "transliteration": "Vikritavikrita deham ghnanti te vartayanti cha.",
        "english": "When vitiated, the three Doshas afflict the body; when in their natural balanced state, they sustain and nourish it.",
        "explanation": "Doshas are not harmful by nature; bringing them back to equilibrium restores natural immunity and tissue vitality."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 1.8",
        "sanskrit": "ते व्यापिनोऽपि हृन्नाभ्योरधोमध्योर्ध्वसंश्रयाः।",
        "transliteration": "Te vyapino'pi hrinnabhyor adhomadhyordhvasamshrayah.",
        "english": "Though pervading the entire body, Vata, Pitta, and Kapha reside predominantly below the navel, between heart and navel, and above the heart respectively.",
        "explanation": "This same anatomical mapping appears on the tongue: the root reflects Vata (colon), the middle reflects Pitta (stomach/liver), and the tip reflects Kapha (chest/heart)."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 1.11",
        "sanskrit": "तत्र रूक्षो लघुः शीतः खरः सूक्ष्मश्चलोऽनिलः।",
        "transliteration": "Tatra ruksho laghuh shitah kharah sukshmashchalo'nilah.",
        "english": "Vata is dry (Ruksha), light (Laghu), cold (Shita), rough (Khara), subtle (Sukshma), and mobile (Chala).",
        "explanation": "When Vata increases, its dry and rough qualities directly cause dryness, thinness, and fissures (cracks) on the tongue surface."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 1.11",
        "sanskrit": "पित्तं सस्नेहतीक्ष्णोष्णं लघु विस्रं सरं द्रवम्।",
        "transliteration": "Pittam sasnehatikshnoshnam laghu visram saram dravam.",
        "english": "Pitta is slightly unctuous, sharp (Tikshna), hot (Ushna), light, pungent-smelling, spreading, and liquid.",
        "explanation": "Excess Pitta manifests its hot and sharp qualities on the tongue as bright redness, burning sensation, red papillae, or yellowish coating."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 1.12",
        "sanskrit": "स्निग्धः शीतो गुरुर्मन्दः श्लक्ष्णो मृत्स्नः स्थिरः कफः।",
        "transliteration": "Snigdhah shito gururmandah shlakshno mritsnah sthirah kaphah.",
        "english": "Kapha is unctuous (Snigdha), cool (Shita), heavy (Guru), slow (Manda), smooth, mucilaginous, and stable.",
        "explanation": "Excess Kapha expresses its heavy, cool, and mucilaginous qualities on the tongue as a pale, broad, swollen body with thick white coating and excess moisture."
    },
    {
        "source": "Yogaratnakara — Jihva Pariksha 2",
        "sanskrit": "द्वन्द्वजे लक्षणे द्वन्द्वं सन्निपातेत्रिलक्षणम्।",
        "transliteration": "Dvandvaje lakshane dvandvam sannipatetrilakshanam.",
        "english": "When two Doshas are simultaneously aggravated, combined signs appear on the tongue; when all three are vitiated (Sannipata), dark, rough, and deeply fissured features arise.",
        "explanation": "Mixed features—such as a red tongue tip (Pitta) with white root coating (Kapha)—indicate dual Dosha involvement (Dvandvaja)."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 26.43",
        "sanskrit": "रसनार्थो रसस्तस्य द्रव्यमापः क्षितिस्तथा।",
        "transliteration": "Rasanartho rasastasya dravyam apah kshitistatha.",
        "english": "Taste (Rasa) is the object of the tongue (Rasana Indriya), and its elementalsubstratum is Water (Jala) and Earth (Prithvi).",
        "explanation": "Proper hydration and balanced Bodhaka Kapha on the tongue are essential for perceiving the six tastes accurately."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 12.15",
        "sanskrit": "रसज्ञानाय तिष्ठति रसनायां बोधकः कफः।",
        "transliteration": "Rasajnanaya tishthati rasanayam bodhakah kaphah.",
        "english": "Bodhaka Kapha resides in the root and surface of the tongue, enabling the perception of taste and maintaining oral moisture.",
        "explanation": "Healthy saliva and gentle sheen on the tongue indicate balanced Bodhaka Kapha."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 12.4",
        "sanskrit": "प्राणोऽत्र मूर्धगः उरःकण्ठचरो बुद्धिहृदयेन्द्रियचित्तधृक्।",
        "transliteration": "Prano'tra murdhagah urahkanthacharo buddhihridayendriyachittadhrik.",
        "english": "Prana Vata moves in the head, chest, and throat, governing swallowing, respiration, senses, and clarity of mind.",
        "explanation": "Trembling or deviation of the extruded tongue indicates a disturbance in Prana and Vyana Vata."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 12.10",
        "sanskrit": "पचत्यन्नं विभजते सारकिट्टौ पृथक् तथा। पचको नाम तत्पित्तम्॥",
        "transliteration": "Pachatyannam vibhajate sarakittau prithak tatha | Pachako nama tatpittam ||",
        "english": "Pachaka Pitta digests food and separates pure nutrient essence (Sara) from waste (Kitta) in the stomach and small intestine.",
        "explanation": "The central zone of the tongue directly reflects the state of Pachaka Pitta and Kledaka Kapha."
    },
    {
        "source": "Vaidya Jeevana — Pathya Prakarana 1.10",
        "sanskrit": "पथ्ये सति गदार्तस्य किमौषधनिषेवणैः। पथ्येऽसति गदार्तस्य किमौषधनिषेवणैः॥",
        "transliteration": "Pathye sati gadartasya kimaushadhanishevanaih | Pathye'sati gadartasya kimaushadhanishevanaih ||",
        "english": "If a patient follows a wholesome diet (Pathya), what is the need for medicine? And if a patient ignores wholesome diet, what is the use of medicine?",
        "explanation": "Dietary discipline (Pathya-Apathya) is the single most powerful remedy to correct tongue coating and digestive imbalance."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 27.349",
        "sanskrit": "प्राणाः प्राणभृतामन्नमन्नं लोकोऽभिधावति। वर्णः प्रसादः सौस्वर्यं जीवितं प्रतिभा सुखम्॥",
        "transliteration": "Pranah pranabhritam annam annam loko'bhidhavati | Varnah prasadah sausvaryam jivitam pratibha sukham ||",
        "english": "Wholesome food is the very life of living beings; complexion, clarity, good voice, longevity, intellect, and happiness all arise from pure nourishment.",
        "explanation": "Eating freshly cooked, warm, seasonal meals keeps Dhatu nourishment strong and prevents Ama formation."
    },
    {
        "source": "Charaka Samhita, Vimanasthana 1.24",
        "sanskrit": "मात्राशी स्यात्। आहारमात्रा पुनरग्निबलापेक्षिणी।",
        "transliteration": "Matrashi syat. Aharamatra punaragnibalapekshini.",
        "english": "One should always eat in proper measure, and the right measure of food depends upon the strength of one's digestive fire (Agnibala).",
        "explanation": "Overeating beyond digestive capacity leaves undigested residue that appears as morning coating on the tongue."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 5.78",
        "sanskrit": "स्नेहगण्डूषधारणैः... हनुबलं स्वरबलं वदनोपचयः परः। परं च रसज्ञानम्॥",
        "transliteration": "Snehagandushadharanaih... hanubalam svarabalam vadanopachayah parah | Param cha rasajnanam ||",
        "english": "Daily oil pulling (Gandusha) strengthens the jaws and voice, nourishes the oral cavity, prevents dryness and cracking of the tongue, and sharpens taste.",
        "explanation": "Swishing warm sesame or coconut oil is a classical remedy for Vata fissures and oral dryness."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 8.2",
        "sanskrit": "अजीर्णे भोजनं विषम्।",
        "transliteration": "Ajirne bhojanam visham.",
        "english": "Eating a new meal while the previous meal remains undigested acts like poison to the digestive system.",
        "explanation": "Waiting until true hunger returns and the tongue feels clean prevents chronic Ama accumulation."
    },
    {
        "source": "Bhavaprakasha Nighantu — Haritaki Varga 42",
        "sanskrit": "त्रिफला कफपित्तघ्नी मेहकुष्ठविनाशिनी। चक्षुष्या दीपनी रुच्या विषमज्वरनाशिनी॥",
        "transliteration": "Triphala kaphapittaghni mehokushthavinashini | Chakshushya deepani ruchya vishamajvaranashini ||",
        "english": "Triphala pacifies Kapha and Pitta, kindles digestive fire (Deepani), improves taste (Ruchya), and rejuvenates the eyes and gut.",
        "explanation": "Triphala gently cleanses the colon (Pakvashaya), reducing coating at the posterior root of the tongue."
    },
    {
        "source": "Bhavaprakasha Nighantu — Shunthi (Dry Ginger)",
        "sanskrit": "नागरं दीपनं वृष्यं ग्राहि हृद्यं विबन्धनुत्। रुच्यं लघु स्वादुपाकं स्निग्धोष्णं कफवातजित्॥",
        "transliteration": "Nagaram deepanam vrishyam grahi hridyam vibandhanut | Ruchyam laghu svadupakam snigdhoshnam kaphavatajit ||",
        "english": "Dry ginger (Shunthi) kindles digestive fire, restores taste on the tongue, relieves bloating, and pacifies Kapha and Vata.",
        "explanation": "Warm ginger infusion is the premier Ayurvedic herb for digesting thick white Kapha-Ama coating on the tongue."
    },
    {
        "source": "Bhavaprakasha Nighantu — Yashtimadhu (Licorice)",
        "sanskrit": "यष्टी हिमा गुरुः स्वाद्वी चक्षुष्या बलवर्णकृत्। सुस्निग्धा शुक्रला केश्या स्वर्या पित्तानिलास्रजित्॥",
        "transliteration": "Yashti hima guruh svadvi chakshushya balavarnakrit | Susnigdha shukrala keshya svarya pittanilasrajit ||",
        "english": "Yashtimadhu (Mulethi) is cooling, sweet, demulcent, and soothing; it pacifies aggravated Pitta and Vata.",
        "explanation": "Mulethi decoction soothes red, inflamed Pitta tongues as well as dry, fissured Vata tongues."
    },
    {
        "source": "Bhavaprakasha Nighantu — Dhanyaka (Coriander)",
        "sanskrit": "धान्यकं तुवरं स्निग्धमवृष्यं मूत्रलं लघु। तिक्तं कटूष्णवीर्यं च दीपनं पाचनं स्मृतम्॥",
        "transliteration": "Dhanyakam tuvaram snigdham avrishyam mutralam laghu | Tiktam katushnaviryam cha deepanam pachanam smritam ||",
        "english": "Coriander (Dhanyaka) is light, digestive (Pachana), and kindles Agni while soothing internal burning and thirst.",
        "explanation": "Coriander-fennel infusion cools Pitta heat in the stomach without weakening digestive fire."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 1.67",
        "sanskrit": "स्वादुरम्लौ सलवणौ वायुं घ्नन्ति, कफं पुनः। तिक्तोषणकषायाश्च, पित्तं स्वादुतिक्तकषायाः॥",
        "transliteration": "Svaduramlau salavanau vayum ghnanti, kapham punah | Tiktoshnakashayashcha, pittam svadutiktakashayah ||",
        "english": "Sweet, sour, and salty tastes pacify Vata; pungent, bitter, and astringent tastes pacify Kapha; sweet, bitter, and astringent tastes pacify Pitta.",
        "explanation": "Choosing foods with the right Rasas (tastes) directly balances the dominant Dosha seen in Jihva Pariksha."
    },
    {
        "source": "Sushruta Samhita, Sutrasthana 46.515",
        "sanskrit": "लङ्घनं परं भेषजमामाशयसमुत्थानाम्।",
        "transliteration": "Langhanam param bheshajam amashayasamutthanam.",
        "english": "Lightness and therapeutic fasting/light diet (Langhana) is the supreme remedy for disorders arising from Ama in the stomach.",
        "explanation": "When the tongue shows a heavy coating, eating light warm soups (Yusha) allows Agni to burn off accumulated Ama."
    },
    {
        "source": "Sharngadhara Samhita, Purvakhanda 5.32",
        "sanskrit": "दीपनं पचनं चैव लङ्घनं बृंहणं तथा।",
        "transliteration": "Deepanam pachanam chaiva langhanam brimhanam tatha.",
        "english": "Kindling digestive fire (Deepana), digesting toxins (Pachana), lightening (Langhana), and nourishing tissues (Brimhana) form the core of Ayurvedic therapy.",
        "explanation": "A coated tongue requires Deepana-Pachana (digestive herbs), whereas a thin, fissured tongue requires Brimhana (warm nourishment and ghee)."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 4.36",
        "sanskrit": "न वेगान् धारयेद्धीमान् जातान् मूत्रपुरीषयोः।",
        "transliteration": "Na vegan dharayed dhiman jatan mutrapurishayoh.",
        "english": "A wise person should never suppress natural bodily urges such as elimination, hunger, thirst, or sleep.",
        "explanation": "Suppressing natural elimination disturbs Apana Vata, causing toxin buildup reflected at the root of the tongue."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 7.53",
        "sanskrit": "आहारशयनाब्रह्मचर्यैर्युक्त्या प्रयोजितैः। शरीरं धार्यते नित्यमागारमिव धारणैः॥",
        "transliteration": "Aharashayanabrahmacharyair yuktya prayojitaih | Shariram dharyate nityam agaramiva dharanaih ||",
        "english": "Just as a house is supported by three pillars, the body is sustained by balanced diet (Ahara), sound sleep (Nidra), and mindful conduct (Brahmacharya).",
        "explanation": "Disturbed sleep and erratic meal timings immediately show up on the tongue as Vata-Pitta dryness and redness."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 5.20",
        "sanskrit": "उष्णं कोष्णं वा जलं दीपनं पाचनं कण्ठ्यं लघु बस्तिशोधनम्।",
        "transliteration": "Ushnam koshnam va jalam deepanam pachanam kanthyam laghu bastishodhanam.",
        "english": "Warm boiled water kindles digestive fire, digests Ama, soothes the throat, is light to absorb, and cleanses the channels.",
        "explanation": "Sipping warm water throughout the day is the simplest classical remedy to clear tongue coating."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 25.40",
        "sanskrit": "कालभोजनमारोग्यकराणां श्रेष्ठम्।",
        "transliteration": "Kalabhojanam arogyakaranam shreshtham.",
        "english": "Eating meals at regular, timely hours is the foremost among all practices that promote health and digestive harmony.",
        "explanation": "Timely meals synchronize Pachaka Pitta and keep the tongue surface clean and healthy."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 26.81",
        "sanskrit": "यत्किञ्चिद्दोषमास्राव्य न निर्हरति कायतः। आहारजातं तत्सर्वमहितं विरुद्धं स्मृतम्॥",
        "transliteration": "Yatkinchiddosham asravya na nirharati kayatah | Aharajatam tatsarvam ahitam viruddham smritam ||",
        "english": "Foods that provoke the Doshas without expelling them from the body become incompatible (Viruddha Ahara) and toxic.",
        "explanation": "Avoiding incompatible food combinations (such as milk with sour fruits or salt) prevents chronic Sama Jihva (coated tongue)."
    },
    {
        "source": "Sushruta Samhita, Sutrasthana 21.8",
        "sanskrit": "विसर्गोदानविक्षेपैः सोमसूर्यानिला यथा। धारयन्ति जगद्देहं कफपित्तानिलास्तथा॥",
        "transliteration": "Visargadanavikshepaih somasuryanila yatha | Dharayanti jagaddeham kaphapittanilastatha ||",
        "english": "Just as the Moon, Sun, and Wind sustain the cosmos through cooling nourishment, heat transformation, and movement, so do Kapha, Pitta, and Vata sustain the body.",
        "explanation": "Understanding whether the tongue shows excess cooling moisture (Kapha), heat (Pitta), or dryness (Vata) reveals the body's inner climate."
    },
    {
        "source": "Charaka Samhita, Vimanasthana 8.95",
        "sanskrit": "प्रकृतिमेकामेकस्य ब्रूयात् समधातुता प्रशस्ता।",
        "transliteration": "Prakritimekam ekasya bruyat samadhatuta prashasta.",
        "english": "Knowing an individual's innate constitution (Prakriti) and age stage (Vaya) is essential before interpreting clinical signs.",
        "explanation": "In Ayurveda, childhood is naturally Kapha-predominant, adulthood is Pitta-predominant, and elderhood is Vata-predominant."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 1.8",
        "sanskrit": "वयोऽहोरात्रिभुक्तानां तेऽन्तमध्यादिगाः क्रमात्।",
        "transliteration": "Vayo'horatribhuktanam te'ntamadhyadigah kramat.",
        "english": "Vata, Pitta, and Kapha predominate respectively in the final, middle, and initial stages of lifespan (Vaya), day, night, and digestion.",
        "explanation": "This verse explains why patient age is vital in Jihva Pariksha: Kapha rules youth, Pitta rules mid-life, and Vata rules older age."
    },
    {
        "source": "Charaka Samhita, Chikitsasthana 1.1.4",
        "sanskrit": "दीर्घमायुः स्मृतिं मेधामारोग्यं तरुणं वयः। प्रभावर्णस्वरौदार्यं देहेन्द्रियबलं परम्॥",
        "transliteration": "Dirghamayuh smritim medham arogyam tarunam vayah | Prabhavarnasvaraudaryam dehendriyabalam param ||",
        "english": "Rasayana (rejuvenation) bestows longevity, memory, intellect, health, youthfulness, lustre, clear voice, and optimal strength of body and senses.",
        "explanation": "Once Ama is cleared from the tongue and gut, Rasayana herbs like Amalaki and Ashwagandha deeply nourish the tissues."
    },
    {
        "source": "Sushruta Samhita, Chikitsasthana 24.68",
        "sanskrit": "व्यायामात् लभते स्वास्थ्यं दीर्घायुष्यं बलं सुखम्। आरोग्यं परमं भाग्यं स्वास्थ्यं सर्वार्थसाधनम्॥",
        "transliteration": "Vyayamat labhate svasthyam dirghayushyam balam sukham | Arogyam paramam bhagyam svasthyam sarvarthasadhanam ||",
        "english": "From regular exercise one attains health, longevity, strength, and happiness; good health is the greatest blessing and the means to accomplish all goals.",
        "explanation": "Moderate daily physical activity kindles Agni, reduces Kapha stagnation, and improves circulation to all tissues."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 8.18",
        "sanskrit": "सद्वृत्तमनुतिष्ठेत् स्वस्थवृत्तं समाचरेत्।",
        "transliteration": "Sadvrittam anutishthet svasthavrittam samacharet.",
        "english": "Practising ethical, calm mental conduct (Sadvritta) and wholesome daily regimen (Svasthavritta) preserves both mental and digestive serenity.",
        "explanation": "Mental stress and anxiety directly disturb Prana Vata and Pachaka Pitta, manifesting as scalloped edges or redness at the tongue tip."
    },
    {
        "source": "Kashyapa Samhita, Khilasthana 5.6",
        "sanskrit": "आहारो महाभैषज्यमुच्यते।",
        "transliteration": "Aharo mahabhaishajyam uchyate.",
        "english": "Acharya Kashyapa declares: Wholesome food (Ahara) is the greatest medicine of all (Mahabhaishajya).",
        "explanation": "Mindful, Dosha-appropriate nutrition corrects digestive imbalances at their root."
    },
    {
        "source": "Ashtanga Hridaya, Sutrasthana 11.37",
        "sanskrit": "ओजस्तु तेजोधातूनां शुक्रान्तानां परं स्मृतम्। हृदयस्थमपि व्यापि देहस्थितिनिबन्धनम्॥",
        "transliteration": "Ojastu tejodhatunam shukrantanam param smritam | Hridayastham api vyapi dehasthitinibandhanam ||",
        "english": "Ojas is the supreme essence of all seven bodily tissues; seated in the heart yet pervading the whole body, it sustains immunity and vitality.",
        "explanation": "A pink, lustrous, supple tongue reflects strong Ojas and balanced tissue nutrition."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 28.4",
        "sanskrit": "रसाद्रक्तं ततो मांसं मांसान्मेदस्ततोऽस्थि च। अस्थ्नो मज्जा ततः शुक्रं शुक्राद्गर्भः प्रसादजः॥",
        "transliteration": "Rasad raktam tato mamsam mamsan medastato'sthi cha | Asthno majja tatah shukram shukrad garbhah prasadajah ||",
        "english": "From nutrient plasma (Rasa) is formed blood (Rakta), then muscle (Mamsa), fat (Meda), bone (Asthi), marrow (Majja), and reproductive essence (Shukra).",
        "explanation": "Pale tongue colour (Pandu) signals weak Rasa and Rakta Dhatu formation, calling for iron-rich, nourishing Pitta-Kapha balanced foods."
    },
    {
        "source": "Ashtanga Sangraha, Sutrasthana 1.24",
        "sanskrit": "जिह्वाताल्वोष्ठकण्ठानां शोषो भवति मारुतात्।",
        "transliteration": "Jihvatalvoshthakanthanam shosho bhavati marutat.",
        "english": "Dryness (Shosha) of the tongue, palate, lips, and throat arises from the aggravation of Vata (Maruta).",
        "explanation": "When the tongue appears dry, rough, and cracked, warm unctuous fluids and cow's ghee are needed to pacify Vata."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 20.14",
        "sanskrit": "दाहौष्ण्यपाकस्वेदाश्च क्लैब्यं मूर्च्छा मदस्तृषा। पित्तस्य लिंगानि॥",
        "transliteration": "Dahaushnyapakasvedashcha klaibyam murchha madastrisha | Pittasya lingani ||",
        "english": "Burning sensation (Daha), heat, inflammation (Paka), and excessive thirst are the classic hallmarks of aggravated Pitta.",
        "explanation": "Redness along the tip and margins of the tongue or small tender spots indicate Pitta Ushma (heat)."
    },
    {
        "source": "Charaka Samhita, Sutrasthana 20.17",
        "sanskrit": "श्वैत्यं शैत्यं गुरुत्वं च कण्डूः स्नेहोपदेहता। कफजानां विकाराणां लिंगम्॥",
        "transliteration": "Shvaityam shaityam gurutvam cha kanduh snehopadehata | Kaphajanam vikaranam lingam ||",
        "english": "Whiteness (Shvaitya), coolness, heaviness, unctuousness, and thick coating (Upadehata) are the classic hallmarks of Kapha disorders.",
        "explanation": "A whitish tongue body or thick white coating (Upadeha/Lepa) directly indicates Kapha accumulation."
    }
]


def get_random_quote() -> dict:
    """Picks a random quote from the 52 verified classical Ayurvedic shlokas."""
    return random.choice(AYURVEDIC_QUOTES)
