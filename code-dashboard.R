
# LIBRARIES
#chargement 
library(shiny)
library(dplyr)
library(ggplot2)
library(plotly)
library(DT)


# extraction des donnees
top_rfm <- read.csv("top_rfm.csv", stringsAsFactors = FALSE)
FP_Growth <- read.csv("FP_Growth.csv", stringsAsFactors = FALSE)

# il nous faut garder seulement les associations utiles
FP_Growth <- FP_Growth %>% filter(!is.na(lift), lift > 1)


# SEGMENTATION DES CLIENTS (RFM)
#règle de décision pour la classification des clients au niveau du RFM score:
#on coupe selon la distribution des clients, en utilisant les quantiles

if(!"segment" %in% colnames(top_rfm)){
  q1 <- quantile(top_rfm$rfm_score_total, 0.33, na.rm = TRUE)
  q2 <- quantile(top_rfm$rfm_score_total, 0.66, na.rm = TRUE)
  
  top_rfm$segment <- cut(top_rfm$rfm_score_total,breaks = c(-Inf, q1, q2, Inf),
labels = c("At Risk","Growth Potential","Loyal"))
}

top_rfm$segment <- as.factor(top_rfm$segment)


############################# KPIs
total_revenue <- sum(top_rfm$monetary, na.rm = TRUE)

segment_summary <- top_rfm %>%
  group_by(segment) %>%
  summarise(customers = n(),revenue = sum(monetary, na.rm = TRUE),
    average_spend = mean(monetary, na.rm = TRUE),
    average_frequency = mean(frequency, na.rm = TRUE),.groups="drop")

rfm_sample <- sample_n(top_rfm, min(15000, nrow(top_rfm)))
fp_display <- head(FP_Growth[order(-FP_Growth$lift), ], 200)


###########onglet BASKET INSIGHTS DATA

FP_Growth <- FP_Growth %>% mutate(rule = paste(antecedents_names, "→", consequents_names))

top_support <- FP_Growth %>% arrange(desc(support)) %>%
distinct(rule, .keep_all = TRUE) %>%
head(4)

top_confidence <- FP_Growth %>% arrange(desc(confidence)) %>%
distinct(rule, .keep_all = TRUE) %>%
head(4)

top_lift <- FP_Growth %>%   arrange(desc(lift)) %>%
distinct(rule, .keep_all = TRUE) %>%
head(4)


######## dashboard

ui <- fluidPage( titlePanel("Data-Driven Retail Insights Dashboard"),
  tabsetPanel(
    tabPanel("Customer Segmentation",
             plotlyOutput("segment_plot"),
             DTOutput("segment_table")
    ),
    
    tabPanel("Customer Behavior",
             plotlyOutput("behavior_plot")
    ),
    
    tabPanel("Market Basket Analysis",
             DTOutput("fp_table")
    ),
    
    tabPanel("Basket Insights",
h3("Top 4 Strongest Product Associations"),
DTOutput("top_support_table"),
h3("Top 4 Rules by Confidence"),
plotlyOutput("confidence_plot"),
h3("Top 4 Most Impactful Associations (Lift)"),
plotlyOutput("lift_plot")
    ),
    
    tabPanel("Business Impact",
             plotlyOutput("revenue_plot")
    ),
    

###################""""
#business simulation onglet

    tabPanel("Business Simulation",
             h3("Targeted Strategy: Grow Revenue + Recover Customers"),
             p("We simulate two retail actions:",
               "1) Cross-selling to Growth Potential customers.",
               "2) Reactivation of At-Risk customers using relevant product suggestions."),
            h4("Growth Potential – Cross-Selling"),
            sliderInput("adoption_growth",
                         "Customers accepting recommendation (%)",
                         min = 0, max = 40, value = 15),
             
             h4("At Risk – Reactivation"),
             sliderInput("reactivation_risk",
                         "Customers returning to buy (%)",
                         min = 0, max = 30, value = 10),
             
             hr(),
             verbatimTextOutput("business_impact")
    )
  )
)


###############""# SERVER

server <- function(input, output, session){
  
  output$top_support_table <- renderDT({
    datatable(top_support[, c("rule","support","confidence","lift")],
              colnames = c("Association Rule","Support","Confidence","Lift"))
  })
  
  output$confidence_plot <- renderPlotly({
    ggplotly(
      ggplot(top_confidence,
             aes(x=reorder(rule,confidence),y=confidence))+
        geom_col(fill="#2C7BE5")+
        coord_flip()
    )
  })
  
  output$lift_plot <- renderPlotly({
    ggplotly(
      ggplot(top_lift,
             aes(x=reorder(rule,lift),y=lift))+
        geom_col(fill="#E5533D")+
        coord_flip()
    )
  })
  
  output$segment_plot <- renderPlotly({
    ggplotly(
      ggplot(segment_summary,aes(segment,customers,fill=segment))+geom_col()
    )
  })
  
  output$segment_table <- renderDT({ datatable(segment_summary) })
  
  output$behavior_plot <- renderPlotly({
    ggplotly(
      ggplot(rfm_sample,aes(frequency,monetary,color=segment))+geom_point(alpha=.4)
    )
  })
  
  output$fp_table <- renderDT({ datatable(fp_display) })
  
  output$revenue_plot <- renderPlotly({
    ggplotly(
      ggplot(segment_summary,aes(segment,revenue,fill=segment))+geom_col()
    )
  })
  

#######BUSINESS SIMULATION 
  
  output$business_impact <- renderText({
    
    #Growth Potential : Cross-Sell ----
growth <- top_rfm %>% filter(segment=="Growth Potential")
    
avg_basket_growth <- mean(growth$monetary / pmax(growth$frequency,1),
      na.rm = TRUE)
    
best_rule <- FP_Growth %>% arrange(desc(support)) %>% slice(1)
    
exposed_growth <- nrow(growth) * best_rule$support
adopting_growth <- exposed_growth * (input$adoption_growth/100)
    
basket_uplift <- 0.08  # +8% panier
revenue_growth <- adopting_growth * avg_basket_growth * basket_uplift
    
    
###At Risk : Reactivation 
 
risk <- top_rfm %>% filter(segment=="At Risk")
avg_basket_risk <- mean(risk$monetary / pmax(risk$frequency,1),na.rm = TRUE)
    
reactivated_clients <- nrow(risk) * (input$reactivation_risk/100)
    
revenue_recovered <- reactivated_clients * avg_basket_risk
    
    
###TOTAL IMPACT ----
    total_impact <- revenue_growth + revenue_recovered
    
    paste(
      "Additional Revenue from Cross-Selling (Growth Potential):",
      round(revenue_growth,0),"€\n",
      
      "Recovered Revenue from Reactivation (At Risk):",
      round(revenue_recovered,0),"€\n",
      
      "--------------------------------------------\n",
      "Total Estimated Business Impact:",
      round(total_impact,0),"€"
    )
  })
}



############run l'aplication dans le terminal
shinyApp(ui,server)


shinyApp(ui, server)


###dans le terminal lancer cette commande
#shiny::runApp()