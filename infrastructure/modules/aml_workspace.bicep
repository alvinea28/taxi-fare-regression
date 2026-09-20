param baseName string
param location string
param stoacctid string
param kvid string
@description('Resource ID of the Application Insights component required by this AML workspace.')
@minLength(1)
param appinsightid string
param crid string
param tags object

// Container Registry may be omitted; Application Insights must be provided.
var hasContainerRegistry = !empty(crid)

// AML workspace
resource amls 'Microsoft.MachineLearningServices/workspaces@2020-09-01-preview' = {
  name: 'mlw-${baseName}'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    tier: 'basic'
    name: 'basic'
  }
  properties: {
    storageAccount: stoacctid
    keyVault: kvid
    applicationInsights: appinsightid
    containerRegistry: hasContainerRegistry ? crid : null
    encryption: {
      status: 'Disabled'
      keyVaultProperties: {
        keyIdentifier: ''
        keyVaultArmId: ''
      }
    }
  }

  tags: tags
}

output amlsName string = amls.name
