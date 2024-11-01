library(MatchIt)
library(dplyr)
library(ggplot2)
library(stats)
library(base)
library(broom)
library(tidyr)
library(readr)
library(dplyr)
library(gtools)
library(data.table)
library(tableone)
options(java.parameters = "-Xmx8000m")
library(rJava)
library(xlsx)
library(readxl)

#wichtig es geht nur wenn es keine fehlenden Werte im Datenset gibt.

Finales_Datenset <- na.omit(Finales_Datenset)
HCC_match<-matchit(HCC~  AGE + SEX + BMI, data=Finales_Datenset, method="nearest", ratio=2, caliper=0.2)
HCC_match10<-matchit(HCC~  AGE + SEX + BMI, data=Finales_Datenset, method="nearest", ratio=10, caliper=0.2)


HCC_gematchedAgeSexBMI <- match.data(HCC_match)
HCC_gematched10AgeSexBMI <- match.data(HCC_match10)


###  matching max. nr, including only pat. with metabolomics
setwd("E:/")
df_metabolomics<- fread("ukb52243.csv", select=c("eid", "23474-0.0","23475-0.0","23476-0.0","23477-0.0","23460-0.0","23479-0.0","23440-0.0","23439-0.0","23441-0.0","23433-0.0","23432-0.0","23431-0.0","23484-0.0","23526-0.0","23561-0.0","23533-0.0","23498-0.0","23568-0.0","23540-0.0","23505-0.0","23575-0.0","23547-0.0","23512-0.0","23554-0.0","23491-0.0","23519-0.0","23580-0.0","23610-0.0","23635-0.0","23615-0.0","23590-0.0","23640-0.0","23620-0.0","23595-0.0","23645-0.0","23625-0.0","23600-0.0","23630-0.0","23585-0.0","23605-0.0","23485-0.0","23418-0.0","23527-0.0","23417-0.0","23562-0.0","23534-0.0","23499-0.0","23569-0.0","23541-0.0","23506-0.0","23576-0.0","23548-0.0","23513-0.0","23416-0.0","23555-0.0","23492-0.0","23520-0.0","23581-0.0","23611-0.0","23636-0.0","23616-0.0","23591-0.0","23641-0.0","23621-0.0","23596-0.0","23646-0.0","23626-0.0","23601-0.0","23631-0.0","23586-0.0","23606-0.0","23473-0.0","23404-0.0","23481-0.0","23430-0.0","23523-0.0","23429-0.0","23558-0.0","23530-0.0","23495-0.0","23565-0.0","23537-0.0","23502-0.0","23572-0.0","23544-0.0","23509-0.0","23428-0.0","23551-0.0","23488-0.0","23516-0.0","23478-0.0","23443-0.0","23450-0.0","23457-0.0","23486-0.0","23422-0.0","23528-0.0","23421-0.0","23563-0.0","23535-0.0","23500-0.0","23570-0.0","23542-0.0","23507-0.0","23577-0.0","23549-0.0","23514-0.0","23420-0.0","23556-0.0","23493-0.0","23521-0.0","23582-0.0","23612-0.0","23637-0.0","23617-0.0","23592-0.0","23642-0.0","23622-0.0","23597-0.0","23647-0.0","23627-0.0","23602-0.0","23632-0.0","23587-0.0","23607-0.0","23470-0.0","23461-0.0","23462-0.0","23480-0.0","23406-0.0","23463-0.0","23465-0.0","23405-0.0","23471-0.0","23466-0.0","23449-0.0","23456-0.0","23447-0.0","23454-0.0","23444-0.0","23451-0.0","23445-0.0","23459-0.0","23452-0.0","23468-0.0","23437-0.0","23434-0.0","23483-0.0","23414-0.0","23525-0.0","23413-0.0","23560-0.0","23532-0.0","23497-0.0","23567-0.0","23539-0.0","23504-0.0","23574-0.0","23546-0.0","23511-0.0","23412-0.0","23553-0.0","23490-0.0","23518-0.0","23579-0.0","23609-0.0","23634-0.0","23614-0.0","23589-0.0","23639-0.0","23619-0.0","23594-0.0","23644-0.0","23624-0.0","23599-0.0","23629-0.0","23584-0.0","23604-0.0","23446-0.0","23458-0.0","23453-0.0","23472-0.0","23402-0.0","23448-0.0","23455-0.0","23438-0.0","23400-0.0","23401-0.0","23436-0.0","23464-0.0","23427-0.0","23415-0.0","23442-0.0","23419-0.0","23482-0.0","23426-0.0","23524-0.0","23425-0.0","23559-0.0","23531-0.0","23496-0.0","23423-0.0","23566-0.0","23538-0.0","23503-0.0","23573-0.0","23545-0.0","23510-0.0","23424-0.0","23552-0.0","23489-0.0","23517-0.0","23411-0.0","23407-0.0","23487-0.0","23410-0.0","23529-0.0","23409-0.0","23564-0.0","23536-0.0","23501-0.0","23571-0.0","23543-0.0","23508-0.0","23578-0.0","23550-0.0","23515-0.0","23408-0.0","23557-0.0","23494-0.0","23522-0.0","23435-0.0","23583-0.0","23613-0.0","23638-0.0","23618-0.0","23593-0.0","23643-0.0","23623-0.0","23598-0.0","23648-0.0","23628-0.0","23603-0.0","23633-0.0","23588-0.0","23608-0.0","23469-0.0","23403-0.0","23467-0.0"))
setwd("C:/Users/Jan/OneDrive/Dokumente/PostDoc/PPSM") 

# create boolean of whether metabolomics are present for patient
df_metabolomics_boolean <- complete.cases(df_metabolomics)
df_metabolomics_boolean <-as.data.frame(df_metabolomics_boolean)
df_metabolomics_boolean$eid <- df_metabolomics$eid

# Entferne NAs, merge with boolean, subset Patienten mit metabolomics
Finales_Datenset <- na.omit(Finales_Datenset)
Finales_Datenset <- merge(Finales_Datenset, df_metabolomics_boolean , by.x = "eid", by.y = "eid", all.x=TRUE)
Pat_with_metabolomics <- subset(Finales_Datenset, df_metabolomics_boolean== TRUE)

#check amount of condition (e.g. HCC) in metabolomics patients (96 HCC -> Match 20 max 2000)
sum(Pat_with_metabolomics$HCC)

# Eigentliches Matching (change ratio for amount of matches for each HCC)
HCC_match20<-matchit(HCC~  AGE + SEX + BMI, data=Pat_with_metabolomics, method="nearest", ratio=20, caliper=0.2)
HCC_gematched20AgeSexBMI <- match.data(HCC_match20)

write.xlsx(HCC_gematched20AgeSexBMI, file="C:/Users/Jan/OneDrive/Dokumente/PostDoc/PPSM", col.names=TRUE, row.names=TRUE, append=FALSE)
