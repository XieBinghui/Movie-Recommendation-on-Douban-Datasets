
library(shiny)
library(networkD3)

ui <- fluidPage(

  sidebarPanel(
    fileInput("file1", "选择文件",
              multiple = TRUE,
              accept = c("text/csv",
                         "text/comma-separated-values,text/plain",
                         ".csv")),
    sliderInput("link_width", "关系线粗浅:",
                min = 0.5, max = 1.5,
                value=0.5, step = 0.1),
    selectInput("link_color", "关系线颜色", 
                choices = c("black", "grey", "red")),
    sliderInput("charge", "节点排斥度:",
                min = -20, max = 5,
                value=-5, step = 1),
    sliderInput("fontsize", "节点名称字体大小:",
                min = 15, max = 25,
                value=20, step = 1),
    selectInput("fontfamily", "节点标签字体", 
                choices = c("宋体", "华文行楷", "黑体"))
    ),
  mainPanel(
    
    forceNetworkOutput("network")
    
  )
)

server <- function(input, output) {
   
  
  
  output$network <- renderForceNetwork({
    
    inFile <- input$file1
    if (is.null(inFile))
      return(NULL)
    hr=read.csv(inFile$datapath)
    hr_edge <- hr[,c('source','target','value')]
    hr_type <- hr[,c('name','group','size')]
    hr_edge <- na.omit(hr_edge)
    hr_type <- na.omit(hr_type)
    forceNetwork(Links = hr_edge,#线性质数据框
                 Nodes = hr_type,#节点性质数据框
                 
                 Source = "target",#连线的源变量
                 Target = "source",#连线的目标变量
                 Value = "value",#连线的粗细值
                 NodeID = "name",#节点名称
                 Group = "group",#节点的分组
                 Nodesize = "size" ,#节点大小，节点数据框中
                 ###美化部分
                 height=1000,
                 linkDistance = JS("function(d){return d.value/10}"),
                 linkWidth = input$link_width,
                 radiusCalculation = JS("d.nodesize"),
                 fontFamily="宋体",#字体设置如"华文行楷" 等
                 fontSize = input$fontsize, #节点文本标签的数字字体大小（以像素为单位）。
                 linkColour=input$link_color,#连线颜色,black,red,blue,  
                 #colourScale ,linkWidth,#节点颜色,red，蓝色blue,cyan,yellow等
                 charge = input$charge,#数值表示节点排斥强度（负值）或吸引力（正值）  
                 opacity = 10,
                 #opacityNoHover = 0.5,
                 legend=T,#显示节点分组的颜色标签
                 arrows=F,#是否带方向
                 bounded=F,#是否启用限制图像的边框
                 #opacityNoHover=1.0,#当鼠标悬停在其上时，节点标签文本的不透明度比例的数值
                 zoom = T)
  })
  
}

shinyApp(ui = ui, server = server)


