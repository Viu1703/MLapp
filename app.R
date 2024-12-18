# Libraries
library(shiny)
library(shinythemes)
library(ggplot2)
library(randomForest)

# Load the pre-trained model
model <- readRDS("rf_model_tuned3.rds")

# Define normal values for each variable
normal_values <- list(
  MemoryComplaints = 0,
  Forgetfulness = 0,
  Disorientation = 0,
  Confusion = 0,
  Depression = 0,
  FamilyHistoryAlzheimers = 0,
  Age = 65,
  BehavioralProblems = 0,
  CardiovascularDisease = 0,
  PhysicalActivity = 8
)

# Define UI for Alzheimer’s prediction with bar charts
ui <- fluidPage(
  theme = shinytheme("darkly"),
  navbarPage(
    "Alzheimer's Detection Tool",
    tabPanel(
      "Detection Using Non-clinical Values",
      sidebarPanel(
        tags$h3("Input:"),
        numericInput("MemoryComplaints", "Memory Complaints (0: No, 1: Yes)", value = 0, min = 0, max = 1),
        numericInput("Forgetfulness", "Forgetfulness (0: No, 1: Yes)", value = 0, min = 0, max = 1),
        numericInput("Disorientation", "Disorientation (0: No, 1: Yes)", value = 0, min = 0, max = 1),
        numericInput("Confusion", "Confusion (0: No, 1: Yes)", value = 0, min = 0, max = 1),
        numericInput("Depression", "Depression (0: No, 1: Yes)", value = 0, min = 0, max = 1),
        numericInput("FamilyHistoryAlzheimers", "Family History of Alzheimer's (0: No, 1: Yes)", value = 0, min = 0, max = 1),
        numericInput("Age", "Age (Years)", value = 60, min = 0, max = 120),
        numericInput("BehavioralProblems", "Behavioral Problems (0: No, 1: Yes)", value = 0, min = 0, max = 1),
        numericInput("CardiovascularDisease", "Cardiovascular Disease (0: No, 1: Yes)", value = 0, min = 0, max = 1),
        numericInput("PhysicalActivity", "Physical Activity (0-10)", value = 5, min = 0, max = 10),
        actionButton("predict", "Predict Alzheimer's Risk")
      ),
      mainPanel(
        h3("Prediction Result"),
        verbatimTextOutput("prediction_output"),
        h4("Comparison with Normal Values (typical scores for a healthy individual)"),
        uiOutput("bar_charts")
      )
    )
  )
)

# Server Logic
server <- function(input, output) {
  
  new_data <- reactive({
    data.frame(
      MemoryComplaints = as.numeric(input$MemoryComplaints),
      Forgetfulness = as.numeric(input$Forgetfulness),
      Disorientation = as.numeric(input$Disorientation),
      Confusion = as.numeric(input$Confusion),
      Depression = as.numeric(input$Depression),
      FamilyHistoryAlzheimers = as.numeric(input$FamilyHistoryAlzheimers),
      Age = as.numeric(input$Age),
      BehavioralProblems = as.numeric(input$BehavioralProblems),
      CardiovascularDisease = as.numeric(input$CardiovascularDisease),
      PhysicalActivity = as.numeric(input$PhysicalActivity)
    )
  })
  
  # Function to check if values are within the valid range
  validate_inputs <- function() {
    if (input$MemoryComplaints < 0 || input$MemoryComplaints > 1)
      showModal(modalDialog("Memory Complaints should be 0 or 1.", easyClose = TRUE))
    else if (input$Forgetfulness < 0 || input$Forgetfulness > 1)
      showModal(modalDialog("Forgetfulness should be 0 or 1.", easyClose = TRUE))
    else if (input$Disorientation < 0 || input$Disorientation > 1)
      showModal(modalDialog("Disorientation should be 0 or 1.", easyClose = TRUE))
    else if (input$Confusion < 0 || input$Confusion > 1)
      showModal(modalDialog("Confusion should be 0 or 1.", easyClose = TRUE))
    else if (input$Depression < 0 || input$Depression > 1)
      showModal(modalDialog("Depression should be 0 or 1.", easyClose = TRUE))
    else if (input$FamilyHistoryAlzheimers < 0 || input$FamilyHistoryAlzheimers > 1)
      showModal(modalDialog("Family History of Alzheimer's should be 0 or 1.", easyClose = TRUE))
    else if (input$Age < 0 || input$Age > 120)
      showModal(modalDialog("Age should be between 0 and 120.", easyClose = TRUE))
    else if (input$BehavioralProblems < 0 || input$BehavioralProblems > 1)
      showModal(modalDialog("Behavioral Problems should be 0 or 1.", easyClose = TRUE))
    else if (input$CardiovascularDisease < 0 || input$CardiovascularDisease > 1)
      showModal(modalDialog("Cardiovascular Disease should be 0 or 1.", easyClose = TRUE))
    else if (input$PhysicalActivity < 0 || input$PhysicalActivity > 10)
      showModal(modalDialog("Physical Activity should be between 0 and 10.", easyClose = TRUE))
    else
      return(TRUE)
    
    return(FALSE)
  }
  
  observeEvent(input$predict, {
    # Validate inputs before proceeding with prediction
    if (!validate_inputs()) return()
    
    # Make a prediction if inputs are valid
    prediction <- predict(model, new_data())
    risk <- ifelse(prediction == 1, "High Risk of Alzheimer's", "Low Risk of Alzheimer's")
    
    output$prediction_output <- renderText({
      paste("The predicted risk of Alzheimer's for the patient is:", risk)
    })
  })
  
  # Render bar charts comparing user values to normal values
  output$bar_charts <- renderUI({
    plot_output_list <- lapply(names(new_data()), function(var) {
      output[[paste0("plot_", var)]] <- renderPlot({
        # Data frame with user input and normal value
        data <- data.frame(
          Category = c("User", "Normal"),
          Value = c(input[[var]], normal_values[[var]])
        )
        
        ggplot(data, aes(x = Category, y = Value, fill = Category)) +
          geom_bar(stat = "identity", position = "dodge", width = 0.4) +
          labs(title = paste(var, "Comparison"), x = var, y = "Value") +
          scale_fill_manual(values = c("User" = "skyblue", "Normal" = "green")) +
          theme_minimal() +
          theme(legend.position = "none")
      })
      plotOutput(paste0("plot_", var), height = "350px")
    })
    
    # Group plots into pairs (2 plots per row)
    plot_pairs <- split(plot_output_list, ceiling(seq_along(plot_output_list) / 2))
    
    # Create a row for each pair of plots
    row_list <- lapply(plot_pairs, function(pair) {
      fluidRow(
        column(6, pair[[1]]),  # First plot in the pair
        column(6, pair[[2]])   # Second plot in the pair
      )
    })
    
    # Combine all rows into a single UI element
    do.call(tagList, row_list)
  })
}

# Run the Shiny App
shinyApp(ui, server)
